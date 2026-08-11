#include "trajectory/motion_planner.hpp"

MotionPlanner::MotionPlanner(AppContext &ctxRef, CommandQueue &commandQueueRef, ControlQueue &controlQueueRef, MotionQueue &motionQueueRef, Robot &robotRef, AudioPlayer &audioRef)
    : ctx(ctxRef), command_queue(commandQueueRef), control_queue(controlQueueRef), motion_queue(motionQueueRef), robot(robotRef),
    behavior_planner(ctxRef, robotRef, audioRef), trajectory_generator(ctxRef, control_queue), motion_log("motion_command") {}

MotionPlanner::~MotionPlanner() {}

void MotionPlanner::run() {
    initialize();

    while (ctx.running.load()) {
        if (auto cmd = command_queue.try_pop()) {
            // 명령이 있으면 motion_queue에 적재
            plan_motions(*cmd);
        }

        // control_queue 잔량이 임계값 이하면 다음 모션 생성
        if (control_queue.size() < threshold) {
            // send_active 이 후 motion_queue가 없으면 모션 채우기
            if (ctx.send_active.load() && motion_queue.empty()) {
                schedule_idle_motion();
            }

            if (auto motion = motion_queue.try_pop()) {
                trajectory_generator.generate_trajectory(*motion);

                if (!ctx.recv_active.load()) ctx.recv_active = true;
                if (!ctx.send_active.load()) ctx.send_active = true;
                if (ctx.play_abort.load()) abort_play_motion();

                record_motion(*motion);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    ctx.running = false;
    std::cerr << "[MotionPlanner] 스레드 종료\n";
}

void MotionPlanner::initialize() {
    behavior_planner.init_poses_from_json();

    // init 포즈 vs 모터 initial_joint_angle 비교
    const auto &init_pose = behavior_planner.poses["init"];
    for (auto &[id, motor] : robot.motors) {
        auto dxl = std::dynamic_pointer_cast<DynamixelMotor>(motor);
        if (dxl) continue;  // 다이나믹셀은 초기 위치 비교 안함

        double diff = std::abs(init_pose[id] - motor->initial_joint_angle);

        if (diff > 1e-4) {
            std::cerr << "[MotionPlanner] init pose mismatch! "
                      << "joint " << id << " (" << motor->name << "): "
                      << "robot_poses.json="  << init_pose[id] * 180.0 / M_PI << "deg  "
                      << "motors.json="       << motor->initial_joint_angle * 180.0 / M_PI << "deg\n";
        }
    }

    trajectory_generator.initialize(behavior_planner.poses);
}

void MotionPlanner::plan_motions(const ParsedCommand& cmd) {
    if (ctx.robot_state.load() == RobotState::SHUTTINGDOWN) return; // 종료 상태가 되면 추가 명령 안받음

    std::vector<MotionPrimitive> motion_sequence = behavior_planner.generate_motion_sequence(cmd);

    int n = motion_sequence.size();
    if (n >= 1) motion_done = false;

    for (int i = 0; i < n; i++) {
        motion_queue.push(motion_sequence[i]);
    }

    record_command(cmd);
}

void MotionPlanner::schedule_idle_motion() {
    if (ctx.robot_state.load() == RobotState::SHUTTINGDOWN) {
        return; // 종료 상태가 되면 추가 명령 안받음
    } else if (ctx.robot_state.load() == RobotState::INIT) {
        // 키 제거하기 전 현재 위치 유지 (고정)
        MotionPrimitive standby_motion;

        standby_motion.type = MotionType::STANDBY;
        motion_queue.push(standby_motion);
    } else if (ctx.robot_state.load() == RobotState::IDLE) {
        // 동작 종료
        if (!motion_done) {
            motion_done = true;
            std::cerr << "[MotionPlanner] 동작을 마쳤습니다.\n";
        }

        // 대기 동작
        MotionPrimitive idle_motion;

        idle_motion.type = MotionType::IDLE;    // IDLE을 MotionType에서 없애고 TRANSLATE(목표 관절각으로 이동)의 반복으로 구현 가능
        motion_queue.push(idle_motion);
    } else if (ctx.robot_state.load() == RobotState::PLAYING) {
        // 연주 종료
        std::cerr << "[MotionPlanner] 연주를 마쳤습니다.\n";
        motion_done = true;

        {
            std::lock_guard<std::mutex> lock(ctx.play_mutex);
            ctx.play_id.clear();    // 연주 끝 -> 현재 곡 없음 (pause_point는 유지)
        }

        // 대기 동작
        ctx.robot_state = RobotState::IDLE;
        MotionPrimitive idle_motion;

        idle_motion.type = MotionType::IDLE;    // IDLE을 MotionType에서 없애고 TRANSLATE(목표 관절각으로 이동)의 반복으로 구현 가능
        motion_queue.push(idle_motion);
    }
}

void MotionPlanner::abort_play_motion() {
    if (ctx.pause_requested.load()) {
        save_pause_point();   // PAUSE: 잔여 모션 폐기 전에 재개 지점 저장
    } else {
        // stop / 내부 에러: 재개 지점 폐기 (RESUME 거부)
        std::lock_guard<std::mutex> lock(ctx.play_mutex);
        ctx.pause_point.valid = false;
    }
    ctx.pause_requested = false;

    motion_queue.clear();     // 남은 PLAYING 전부 폐기
    MotionPrimitive end; end.type = MotionType::DRUM; end.flag = PlayFlag::END;
    motion_queue.push(end);   // END(ready 복귀)만 남김

    ctx.play_abort = false;
    std::cerr << "[MotionPlanner] 연주가 비정상 종료됩니다.\n";
}

// 중단 시점의 재개 지점 저장.
// MotionQueue 맨 앞은 아직 궤적을 생성하지 않은 primitive이므로,
// 그 안의 "다음에 칠 이벤트"의 마디 번호가 곧 재개 지점이다.
// 반드시 motion_queue.clear() 전에 불러야 한다.
void MotionPlanner::save_pause_point() {
    std::optional<MotionPrimitive> front_motion = motion_queue.try_peek();

    std::lock_guard<std::mutex> lock(ctx.play_mutex);
    ctx.pause_point.valid = false;

    if (!front_motion.has_value()) return;      // 큐가 비어 있음 (곡이 사실상 끝난 상태)
    if (ctx.play_id.empty()) return;            // 연주 중인 곡을 모름

    const MotionPrimitive &motion = front_motion.value();
    if (motion.type != MotionType::DRUM) return;
    if (motion.flag != PlayFlag::PLAYING) return;           // START/END는 재개 지점이 없음
    if (motion.robotic_drum_score.size() < 2) return;

    // robotic_drum_score는 악보 줄(DrumEvent) window:
    // [0]은 직전 이벤트(이미 타격 완료), [1]이 아직 안 친 첫 이벤트
    ctx.pause_point.play_id = ctx.play_id;
    ctx.pause_point.bar = static_cast<int>(motion.robotic_drum_score[1].bar);
    ctx.pause_point.valid = true;

    std::cerr << "[MotionPlanner] 재개 지점 저장: id=" << ctx.pause_point.play_id
              << ", bar=" << ctx.pause_point.bar << "\n";
}

// ===== log =====
void MotionPlanner::record_command(const ParsedCommand& cmd) {
    // Opcode -> 문자열
    auto opcode_to_string = [](Opcode op) -> std::string {
        switch (op) {
            case Opcode::LOOK:    return "LOOK";
            case Opcode::GESTURE: return "GESTURE";
            case Opcode::MOVE:    return "MOVE";
            case Opcode::POSE:    return "POSE";
            case Opcode::HIT:     return "HIT";
            case Opcode::PLAY:    return "PLAY";
            case Opcode::PLAY_CTRL: return "PLAY_CTRL";
            case Opcode::START:   return "START";
            case Opcode::READY:   return "READY";
            case Opcode::PAUSE:   return "PAUSE";
            case Opcode::RESUME:  return "RESUME";
            case Opcode::GET_STATUS: return "GET_STATUS";
            case Opcode::QUIT:    return "QUIT";
            case Opcode::UNKNOWN: return "UNKNOWN";
            default:              return "UNKNOWN";
        }
    };

    std::vector<std::string> log = {"CMD"};

    log.push_back(opcode_to_string(cmd.opcode));
    log.push_back(cmd.valid ? "valid" : "invalid");

    // 인자 (OPCODE|arg1|arg2... 형태 그대로)
    for (const auto& arg : cmd.args) {
        log.push_back(arg);
    }

    motion_log.record(log);
}

void MotionPlanner::record_motion(const MotionPrimitive& motion) {
    auto profile_to_string = [](TrajectoryProfile p) -> std::string {
        switch (p) {
            case TrajectoryProfile::TRAPEZOIDAL: return "trapezoidal";
            case TrajectoryProfile::CUBIC:       return "cubic";
            case TrajectoryProfile::QUINTIC:     return "quintic";
            case TrajectoryProfile::COSINE:      return "cosine";
            default:                             return "?";
        }
    };
    auto space_to_string = [](TrajectorySpace s) -> std::string {
        return s == TrajectorySpace::JOINT ? "joint" : "task";
    };
    auto flag_to_string = [](PlayFlag f) -> std::string {
        switch (f) {
            case PlayFlag::START:   return "start";
            case PlayFlag::PLAYING: return "playing";
            case PlayFlag::END:     return "end";
            default:                return "?";
        }
    };

    std::vector<std::string> log = {"MOTION"};

    switch (motion.type) {
        case MotionType::IDLE:
            log.push_back("idle");
            break;

        case MotionType::STANDBY:
            log.push_back("standby");
            break;

        case MotionType::TRANSLATE:
            log.push_back("translate");
            log.push_back(space_to_string(motion.space));
            log.push_back(profile_to_string(motion.profile));
            log.push_back("t_total=" + std::to_string(motion.t_total));
            if (motion.space == TrajectorySpace::JOINT) {
                log.push_back("q_target_n=" + std::to_string(motion.q_target.size()));
                for (double q : motion.q_target) {
                    log.push_back(std::to_string(q));
                }
            } else {
                // task space: 오른팔/왼팔 목표 좌표
                std::string pr = "R(";
                for (size_t i = 0; i < motion.p_target_R.size(); ++i) {
                    pr += std::to_string(motion.p_target_R[i]);
                    if (i + 1 < motion.p_target_R.size()) pr += ",";
                }
                pr += ")";
                std::string pl = "L(";
                for (size_t i = 0; i < motion.p_target_L.size(); ++i) {
                    pl += std::to_string(motion.p_target_L[i]);
                    if (i + 1 < motion.p_target_L.size()) pl += ",";
                }
                pl += ")";
                log.push_back(pr);
                log.push_back(pl);
            }
            break;

        case MotionType::DRUM:
            log.push_back("drum");
            log.push_back(flag_to_string(motion.flag));
            log.push_back("events=" + std::to_string(motion.robotic_drum_score.size()));
            // 구간의 시간 범위 (첫/마지막 이벤트 누적 시간)
            if (!motion.robotic_drum_score.empty()) {
                log.push_back("t_first=" + std::to_string(motion.robotic_drum_score.front().t));
                log.push_back("t_last="  + std::to_string(motion.robotic_drum_score.back().t));
                log.push_back("bar_first=" + std::to_string(motion.robotic_drum_score.front().bar));
                log.push_back("bar_last="  + std::to_string(motion.robotic_drum_score.back().bar));
            } else if (flag_to_string(motion.flag) == "start") {
                log.push_back("init_note_r="  + std::to_string(motion.init_note_r));
                log.push_back("init_note_l="  + std::to_string(motion.init_note_l));
            }
            break;

        case MotionType::REPLAY:
            log.push_back("replay");
            log.push_back(motion.replay_csv);
            break;
    }

    motion_log.record(log);
}
