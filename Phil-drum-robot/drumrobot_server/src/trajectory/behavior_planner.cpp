#include "trajectory/behavior_planner.hpp"

// 관절 ID 상수 (motors.json 참조)
namespace JointID {
    constexpr int WAIST            = 0;
    constexpr int R_SHOULDER_1     = 1;
    constexpr int L_SHOULDER_1     = 2;
    constexpr int R_SHOULDER_2     = 3;
    constexpr int R_ELBOW          = 4;
    constexpr int L_SHOULDER_2     = 5;
    constexpr int L_ELBOW          = 6;
    constexpr int R_WRIST          = 7;
    constexpr int L_WRIST          = 8;
    constexpr int R_PEDAL          = 9;
    constexpr int L_PEDAL          = 10;
    constexpr int HEAD_YAW         = 11;
    constexpr int HEAD_PITCH       = 12;
}

BehaviorPlanner::BehaviorPlanner(AppContext &ctxRef, Robot &robotRef, AudioPlayer &audioRef)
    : ctx(ctxRef), robot(robotRef), audio_player(audioRef) {
    // 초기 자세를 last_q_target으로 설정 (모터의 initial_joint_angle 사용)
    last_q_target.resize(ROBOT::NUM_JOINT, 0.0);
    for (const auto &[id, motor] : robot.motors) {
        if (id < ROBOT::NUM_JOINT) {
            last_q_target[id] = motor->initial_joint_angle;
        }
    }

    init_play_list_from_json();
}

BehaviorPlanner::~BehaviorPlanner() {

}

std::vector<MotionPrimitive> BehaviorPlanner::generate_motion_sequence(const ParsedCommand& parsed) {
    std::vector<MotionPrimitive> sequence;

    if (!parsed.valid) {
        std::cerr << "[BehaviorPlanner] Invalid command\n";
        return sequence;
    }

    Opcode opcode = parsed.opcode;

    // ===== send_active 전 =====
    // 시작/종료 명령만 처리
    if (!ctx.send_active.load()) {
        if (opcode == Opcode::START) {
            return handle_start();
        } else if (opcode == Opcode::QUIT) {
            ctx.robot_state = RobotState::SHUTTINGDOWN;
            return sequence;
        } else {
            std::cerr << "[BehaviorPlanner] 수행할 수 없는 명령 (send_active=false): opcode="
                      << static_cast<int>(opcode) << "\n";
            return sequence;
        }
    }

    // ===== send_active 후 =====
    switch (opcode) {
        case Opcode::READY: {
            handle_ready();
            return sequence;
        }
        case Opcode::LOOK:    return handle_look(parsed.args);
        case Opcode::GESTURE: return handle_gesture(parsed.args);
        case Opcode::MOVE:    return handle_move(parsed.args);
        case Opcode::POSE:    return handle_pose(parsed.args);
        case Opcode::HIT:     return handle_hit(parsed.args);
        case Opcode::PLAY:    return handle_play(parsed.args);
        case Opcode::REPLAY:  return handle_replay(parsed.args);
        case Opcode::PAUSE: {
            handle_pause();
            return sequence;
        }
        case Opcode::RESUME:  return handle_resume();
        case Opcode::PLAY_CTRL: {
            handle_play_ctrl(parsed.args);
            return sequence;
        }
        case Opcode::QUIT:    return handle_quit();
        case Opcode::START:
            std::cerr << "[BehaviorPlanner] 이미 시작된 상태\n";
            return sequence;
        default:
            std::cerr << "[BehaviorPlanner] Unknown opcode\n";
            return sequence;
    }
}

void BehaviorPlanner::init_poses_from_json() {
    using json = nlohmann::json;

    std::ifstream f("drumrobot_server/config/robot_poses.json");
    if (!f.is_open()) {
        std::cerr << "[BehaviorPlanner] Failed to open config/robot_poses.json\n";
        return;
    }
    json config = json::parse(f);

    for (auto &[name, angles] : config["poses"].items()) {
        for (auto &a : angles) {
            poses[name].push_back(a.get<double>() * M_PI / 180.0);
        }
    }
}

void BehaviorPlanner::init_play_list_from_json() {
    using json = nlohmann::json;

    std::ifstream f("drumrobot_server/config/play_list.json");
    if (!f.is_open()) {
        std::cerr << "[BehaviorPlanner] Failed to open config/play_list.json\n";
        return;
    }
    json config = json::parse(f);

    for (auto &[id, entry] : config["play_list"].items()) {
        PlayEntry e;
        e.score = entry.value("score", "");
        e.audio = entry.value("audio", "");
        e.init_note_r = entry.value("init_note_r", 1);
        e.init_note_l = entry.value("init_note_l", 1);

        play_list[id] = e;
    }
}

// =============================================================
// Opcode별 핸들러
// =============================================================

// START: home 포즈로 이동
std::vector<MotionPrimitive> BehaviorPlanner::handle_start() {
    std::vector<MotionPrimitive> sequence;

    auto it = poses.find("home");
    if (it == poses.end()) {
        std::cerr << "[BehaviorPlanner] 'home' pose not found in robot_poses.json\n";
        return sequence;
    }

    sequence.push_back(make_translate(it->second, DEFAULT_MOVE_TIME));
    set_last_q_target(it->second);

    std::cout   << "\n========================================\n"
                << " 모터 토크 ON\n"
                << " 1. 고정 키를 모두 제거하세요.\n"
                << " 2. 제거 후 'READY' 명령을 입력하세요.\n"
                << "========================================\n\n";
    ctx.robot_state = RobotState::INIT;

    return sequence;
}

// READY: idle state로 변경
void BehaviorPlanner::handle_ready() {
    if (ctx.robot_state.load() == RobotState::INIT) {
        ctx.robot_state = RobotState::IDLE;
    }
}

// LOOK pan tilt : 머리 yaw, pitch 제어
std::vector<MotionPrimitive> BehaviorPlanner::handle_look(const std::vector<std::string>& args) {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] LOOK rejected: only allowed in IDLE\n";
        return sequence;
    }

    try {
        double pan_deg  = std::stod(args[0]);
        double tilt_deg = std::stod(args[1]);

        // 마지막 목표를 복사해서 head 관절만 갱신
        std::vector<double> q_target = last_q_target;
        q_target[JointID::HEAD_YAW]   = deg_to_rad(pan_deg);
        q_target[JointID::HEAD_PITCH] = deg_to_rad(tilt_deg);

        sequence.push_back(make_translate(q_target, LOOK_MOVE_TIME));
        set_last_q_target(q_target);
    } catch (const std::exception &e) {
        std::cerr << "[BehaviorPlanner] LOOK parsing error: " << e.what() << "\n";
    }

    return sequence;
}

// GESTURE type : 미리 정의된 제스처 시퀀스
std::vector<MotionPrimitive> BehaviorPlanner::handle_gesture(const std::vector<std::string>& args) {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] GESTURE rejected: only allowed in IDLE\n";
        return sequence;
    }

    const std::string& type = args[0];

    if (type == "nod") {
        // 끄덕임: 아래 → 위 → 정면
        std::vector<double> q;
        q = last_q_target; q[JointID::HEAD_PITCH] = deg_to_rad(20.0);
        sequence.push_back(make_translate(q, GESTURE_MOVE_TIME));
        q[JointID::HEAD_PITCH] = deg_to_rad(-20.0);
        sequence.push_back(make_translate(q, GESTURE_MOVE_TIME));
        q[JointID::HEAD_PITCH] = 0.0;
        sequence.push_back(make_translate(q, GESTURE_MOVE_TIME));
        set_last_q_target(q);
    }
    else if (type == "shake") {
        // 도리도리: 좌 → 우 → 정면
        std::vector<double> q;
        q = last_q_target; q[JointID::HEAD_YAW] = deg_to_rad(30.0);
        sequence.push_back(make_translate(q, GESTURE_MOVE_TIME));
        q[JointID::HEAD_YAW] = deg_to_rad(-30.0);
        sequence.push_back(make_translate(q, GESTURE_MOVE_TIME));
        q[JointID::HEAD_YAW] = 0.0;
        sequence.push_back(make_translate(q, GESTURE_MOVE_TIME));
        set_last_q_target(q);
    }
    else if (type == "wave" || type == "hi") {
        // 인사: 오른팔 들기 + 손목 흔들기
        // 1) 오른팔 인사 자세
        std::vector<double> q = last_q_target;
        q[JointID::R_SHOULDER_1] = deg_to_rad(45.0);
        q[JointID::R_SHOULDER_2] = deg_to_rad(45.0);
        q[JointID::R_ELBOW]      = deg_to_rad(90.0);
        q[JointID::R_WRIST]      = 0.0;
        q[JointID::HEAD_YAW]     = deg_to_rad(-20.0);
        q[JointID::HEAD_PITCH]   = deg_to_rad(-5.0);
        sequence.push_back(make_translate(q, DEFAULT_MOVE_TIME));

        // 2) 손목 좌우 흔들기 3회
        for (int i = 0; i < 3; i++) {
            q[JointID::R_WRIST] = deg_to_rad(25.0);
            sequence.push_back(make_translate(q, 0.4));
            q[JointID::R_WRIST] = deg_to_rad(-25.0);
            sequence.push_back(make_translate(q, 0.4));
        }
        // 복귀
        q[JointID::R_WRIST] = 0.0;
        sequence.push_back(make_translate(q, 0.4));
        set_last_q_target(q);
    }
    else if (type == "hurray" || type == "happy") {
        // 환호: 양팔 들기
        std::vector<double> q = last_q_target;
        q[JointID::R_SHOULDER_1] = deg_to_rad(60.0);
        q[JointID::L_SHOULDER_1] = deg_to_rad(120.0);
        q[JointID::R_SHOULDER_2] = deg_to_rad(65.0);
        q[JointID::L_SHOULDER_2] = deg_to_rad(65.0);
        q[JointID::R_ELBOW]      = deg_to_rad(95.0);
        q[JointID::L_ELBOW]      = deg_to_rad(95.0);
        q[JointID::R_WRIST]      = 0.0;
        q[JointID::L_WRIST]      = 0.0;
        q[JointID::HEAD_PITCH]   = deg_to_rad(-15.0);
        sequence.push_back(make_translate(q, DEFAULT_MOVE_TIME));
        set_last_q_target(q);
    }
    else {
        std::cerr << "[BehaviorPlanner] Unknown gesture: " << type << "\n";
    }

    return sequence;
}

// MOVE: [motor_name, angle_deg] [move_time]
std::vector<MotionPrimitive> BehaviorPlanner::handle_move(const std::vector<std::string>& args) {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] MOVE rejected: only allowed in IDLE\n";
        return sequence;
    }

    if (args.empty()) {
        std::cerr << "[BehaviorPlanner] MOVE rejected: no arguments\n";
        return sequence;
    }

    try {
        std::vector<double> q_target = last_q_target;
        double move_time = DEFAULT_MOVE_TIME;
        size_t i = 0;
        bool any_applied = false;

        // (motor_name, angle_deg) 쌍을 순회하며 적용
        while (i + 1 < args.size()) {
            const std::string& motor_name = args[i];
            int motor_id = find_motor_id(motor_name);
            if (motor_id < 0) {
                std::cerr << "[BehaviorPlanner] Unknown motor name: " << motor_name << "\n";
                return sequence;  // 하나라도 잘못되면 전체 취소
            }

            double angle_deg = std::stod(args[i + 1]);
            q_target[motor_id] = deg_to_rad(angle_deg);
            any_applied = true;
            i += 2;
        }

        // 마지막에 홀수로 남은 인자가 있으면 move_time으로 해석
        if (i < args.size()) {
            move_time = std::stod(args[i]);
        }

        if (!any_applied) {
            std::cerr << "[BehaviorPlanner] MOVE rejected: no valid motor/angle pairs\n";
            return sequence;
        }

        sequence.push_back(make_translate(q_target, move_time));
        set_last_q_target(q_target);
    } catch (const std::exception &e) {
        std::cerr << "[BehaviorPlanner] MOVE parsing error: " << e.what() << "\n";
    }

    return sequence;
}

// POSE pose_name : 사전 정의 포즈로 이동
std::vector<MotionPrimitive> BehaviorPlanner::handle_pose(const std::vector<std::string>& args) {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] POSE rejected: only allowed in IDLE\n";
        return sequence;
    }

    const std::string& pose_name = args[0];

    auto it = poses.find(pose_name);
    if (it == poses.end()) {
        std::cerr << "[BehaviorPlanner] Unknown pose: " << pose_name << "\n";
        return sequence;
    }

    sequence.push_back(make_translate(it->second, DEFAULT_MOVE_TIME, TrajectoryProfile::TRAPEZOIDAL));
    set_last_q_target(it->second);

    // shutdown 포즈로 이동하는 경우 종료 플래그 세팅
    if (pose_name == "shutdown") {
        ctx.robot_state = RobotState::SHUTTINGDOWN;
    }

    return sequence;
}

// HIT target : 드럼 타격
std::vector<MotionPrimitive> BehaviorPlanner::handle_hit(const std::vector<std::string>& args) {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] HIT rejected: only allowed in IDLE\n";
        return sequence;
    }

    const std::string& target = args[0];

    if (instrument_name_to_id.find(target) != instrument_name_to_id.end()) {
        MotionPrimitive start; start.type = MotionType::DRUM; start.flag = PlayFlag::START;
        sequence.push_back(start);

        int id = instrument_name_to_id.at(target);
        sequence.push_back(make_drum_hit(DEFAULT_HIT_TIME, id));

        MotionPrimitive end; end.type = MotionType::DRUM; end.flag = PlayFlag::END;
        sequence.push_back(end);

        // 드럼 모션은 항상 ready 포즈에서 시작해 ready 포즈로 복귀한다.
        // 따라서 타격 종료 후의 관절각은 ready 포즈와 같다.
        // NOTE: 추후 드럼 모션의 종료 자세가 동적으로 바뀌면,
        //       여기서 드럼 모션 생성기가 산출한 실제 마지막 q_target으로 갱신해야 함.
        auto ready_it = poses.find("ready");
        if (ready_it != poses.end()) {
            set_last_q_target(ready_it->second);
        } else {
            std::cerr << "[BehaviorPlanner] HIT: 'ready' pose not found; last_q_target 미갱신\n";
        }
    } else {
        std::cerr << "[BehaviorPlanner] Unknown target instrument: " << target << "\n";
        return sequence;
    }

    return sequence;
}

// PLAY score_name : 드럼 연주
std::vector<MotionPrimitive> BehaviorPlanner::handle_play(const std::vector<std::string>& args) {
    return make_play_sequence(args[0], 0);
}

// REPLAY csv_path : sim에서 뽑은 관절궤적 CSV를 오픈루프로 재생 (Stage A 검증용)
//   실시간 정책(closed-loop) 대체 전, "미리 뽑은 궤적을 로봇이 잘 따라가는지" 확인하는 경로.
//   IDLE에서만 허용. 실제 CSV 파싱/보간/스트리밍은 TrajectoryGenerator::generate_replay_trajectory 가 담당.
std::vector<MotionPrimitive> BehaviorPlanner::handle_replay(const std::vector<std::string>& args) {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] REPLAY rejected: only allowed in IDLE\n";
        return sequence;
    }

    MotionPrimitive m;
    m.type = MotionType::REPLAY;
    m.replay_csv = args[0];
    sequence.push_back(m);
    return sequence;
}

// PAUSE : 연주 일시정지. 재개 지점을 저장하는 abort 경로로 보낸다.
void BehaviorPlanner::handle_pause() {
    if (ctx.robot_state.load() != RobotState::PLAYING) {
        std::cerr << "[BehaviorPlanner] PAUSE rejected: only allowed in PLAYING\n";
        return;
    }

    ctx.pause_requested = true;
    ctx.play_abort = true;
    std::cerr << "[BehaviorPlanner] 일시정지 요청 -> 재개 지점 저장 후 ready 복귀\n";
}

// RESUME : 저장된 재개 지점부터 다시 연주
std::vector<MotionPrimitive> BehaviorPlanner::handle_resume() {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] RESUME rejected: only allowed in IDLE\n";
        return sequence;
    }

    std::string resume_id;
    int resume_bar = 0;
    {   // play_mutex는 이 블록 안에서만 잡는다
        std::lock_guard<std::mutex> lock(ctx.play_mutex);
        if (!ctx.pause_point.valid) {
            std::cerr << "[BehaviorPlanner] RESUME rejected: 저장된 재개 지점이 없습니다\n";
            return sequence;
        }
        resume_id = ctx.pause_point.play_id;
        resume_bar = ctx.pause_point.bar;
    }

    std::cerr << "[BehaviorPlanner] 재개: id=" << resume_id << ", bar=" << resume_bar << "\n";
    return make_play_sequence(resume_id, resume_bar);
}

// 재개 구간에서 해당 손의 첫 타격 악기 번호를 찾는다. 없으면 0.
static int find_first_note(const std::vector<DrumEvent>& rds, bool is_right) {
    for (size_t i = 1; i < rds.size(); i++) {
        int note = is_right ? rds[i].note_num_R : rds[i].note_num_L;
        if (note == 5 && !rds[i].is_closed_hihat) {
            note = 9;   // 오픈 하이햇
        }
        if (note != 0) {
            return note;
        }
    }
    return 0;
}

// 악보를 읽어 연주 모션 시퀀스를 만든다. start_bar > 0 이면 그 마디부터 시작(재개).
std::vector<MotionPrimitive> BehaviorPlanner::make_play_sequence(const std::string& id, int start_bar) {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] PLAY rejected: only allowed in IDLE\n";
        return sequence;
    }

    auto it = play_list.find(id);
    if (it == play_list.end()) {
        std::cerr << "[BehaviorPlanner] PLAY: 알 수 없는 id: " << id << "\n";
        return sequence;
    }
    const std::string& score_name = it->second.score;
    const std::string& audio_name = it->second.audio;

    std::ifstream inputFile;
    std::string score_path = "drumrobot_server/data/scores/" + score_name + ".txt";
    inputFile.open(score_path);

    if (!inputFile.is_open()) {
        std::cerr << "[BehaviorPlanner] PLAY: 악보 파일을 열 수 없습니다: " << score_path << "\n";
        return sequence;
    }

    if (start_bar > 0) {
        audio_player.clear_track();     // 재개는 무음 (음악 중간부터 재생은 미지원)
    } else {
        audio_player.set_track(audio_name);
    }

    std::vector<DrumEvent> rds;
    DrumEvent Dummy;
    rds.push_back(Dummy);   // rds[0]
    int start_idx = 0, end_idx = 0;

    double bpm = 100.0;
    double last_t = 0.0;

    MotionPrimitive start; start.type = MotionType::DRUM; start.flag = PlayFlag::START;
    start.init_note_r = it->second.init_note_r;
    start.init_note_l = it->second.init_note_l;
    sequence.push_back(start);

    std::string row;
    while (getline(inputFile, row)) {
        istringstream iss(row);
        std::string item;
        std::vector<std::string> items;
        
        while (getline(iss, item, '\t')) {
            item = trim_whitespace(item);
            items.push_back(item);
        }

        if (items.empty() || items[0].empty()) {
            continue;   // 빈 줄 무시
        }

        if (items[0] == "bpm") {
            if (items.size() < 2) continue;
            try { bpm = std::stod(items[1]); } catch (...) { continue; }
        } else if (items[0] == "end") {
            while (start_idx < end_idx) {
                sequence.push_back(make_drum_play(std::vector<DrumEvent>(rds.begin() + start_idx, rds.end()))); // rds.end() == rds.begin() + end_idx + 1
                start_idx++;
            }
            break;
        } else {
            DrumEvent ev;
            if (!make_drum_event(items, bpm, last_t, ev)) break;
            if (static_cast<int>(ev.bar) < start_bar) {
                continue;   // 재개 지점 이전 줄은 건너뜀 (bpm 줄은 위에서 계속 적용, last_t 미누적)
            }
            rds.push_back(ev);

            end_idx++;
            last_t = rds[end_idx].t;

            // 2.4s : 100bpm 기준 한 마디 시간
            if ((rds[end_idx].t - rds[start_idx].t) * bpm / 100.0 >= 2.4) {
                sequence.push_back(make_drum_play(std::vector<DrumEvent>(rds.begin() + start_idx, rds.begin() + end_idx + 1)));
                start_idx++;
            }
        }
    }
    inputFile.close();

    if (start_bar > 0 && rds.size() < 2) {
        std::cerr << "[BehaviorPlanner] PLAY: 재개 마디(" << start_bar
                  << ")가 악보 범위 밖입니다. 연주 없이 ready로 복귀합니다\n";
    }

    if (start_bar > 0 && rds.size() >= 2) {
        // 준비 자세는 타격 없이 악기 위에 대기해야 하므로, 재개 시 init_note를
        // 곡 도입부 설정값이 아닌 재개 구간의 첫 타격 악기로 바꾼다.
        int first_note_r = find_first_note(rds, true);
        int first_note_l = find_first_note(rds, false);
        if (first_note_r != 0) sequence[0].init_note_r = first_note_r;
        if (first_note_l != 0) sequence[0].init_note_l = first_note_l;
        std::cerr << "[BehaviorPlanner] 재개 준비 위치: R=" << sequence[0].init_note_r
                  << ", L=" << sequence[0].init_note_l << "\n";
    }

    MotionPrimitive end; end.type = MotionType::DRUM; end.flag = PlayFlag::END;
    sequence.push_back(end);

    // 드럼 연주는 항상 ready 포즈에서 시작해 ready 포즈로 복귀한다.
    // 따라서 연주 종료 후의 관절각은 ready 포즈와 같다.
    // NOTE: 추후 연주 모션의 종료 자세가 동적으로 바뀌면,
    //       여기서 드럼 모션 생성기가 산출한 실제 마지막 q_target으로 갱신해야 함.
    auto ready_it = poses.find("ready");
    if (ready_it != poses.end()) {
        set_last_q_target(ready_it->second);
    } else {
        std::cerr << "[BehaviorPlanner] PLAY: 'ready' pose not found; last_q_target 미갱신\n";
    }

    {   // play_mutex는 이 블록 안에서만 잡는다 (lock_guard가 '}'에서 unlock)
        std::lock_guard<std::mutex> lock(ctx.play_mutex);
        ctx.play_id = id;
        ctx.pause_point.valid = false;      // 새 연주 시작 -> 이전 재개 지점 폐기
    }

    ctx.robot_state = RobotState::PLAYING;
    ctx.play_speed_scale = 1.0;
    return sequence;
}

// PLAY_CTRL 드럼 연주 제어
void BehaviorPlanner::handle_play_ctrl(const std::vector<std::string>& args) {
    if (ctx.robot_state.load() != RobotState::PLAYING) {
        std::cerr << "[BehaviorPlanner] PLAY_CTRL rejected: only allowed in PLAYING\n";
        return;
    }

    const std::string& ctrl = args[0];

    if (ctrl == "stop") {
        ctx.pause_requested = false;    // stop은 재개 지점을 남기지 않는다
        ctx.play_abort = true;
        std::cerr << "[BehaviorPlanner] 연주 중지 요청 -> 잔여 모션 폐기 후 ready 복귀\n";
    }
    else if (ctrl == "speed") {
        if (args.size() < 2) {
            std::cerr << "[BehaviorPlanner] PLAY_CTRL speed: 배율 인자가 없습니다\n";
            return;
        }
        double scale;
        try {
            scale = std::stod(args[1]);
        } catch (const std::exception& e) {
            std::cerr << "[BehaviorPlanner] PLAY_CTRL speed: 잘못된 배율 값: " << args[1] << "\n";
            return;
        }
        double clamped = std::clamp(scale, MIN_SCALE, MAX_SCALE);
        ctx.play_speed_scale = clamped;
        std::cerr << "[BehaviorPlanner] 연주 속도 배율: " << clamped << "x";
        if (clamped != scale) {
            std::cerr << " (요청 " << scale << " 가 [" << MIN_SCALE << ", " << MAX_SCALE << "] 로 제한됨)";
        }
        std::cerr << "\n";
    }
    else {
        std::cerr << "[BehaviorPlanner] Unknown PLAY_CTRL: " << ctrl << "\n";
    }
}

std::vector<MotionPrimitive> BehaviorPlanner::handle_quit() {
    std::vector<MotionPrimitive> sequence;
    if (ctx.robot_state.load() != RobotState::IDLE) {
        std::cerr << "[BehaviorPlanner] QUIT rejected: only allowed in IDLE\n";
        return sequence;
    }

    // shutdown 포즈로 이동 후 종료 플래그 세팅
    auto it = poses.find("shutdown");
    if (it != poses.end()) {
        sequence.push_back(make_translate(it->second, DEFAULT_MOVE_TIME));
        set_last_q_target(it->second);
    }
    ctx.robot_state = RobotState::SHUTTINGDOWN;
    return sequence;
}

// =============================================================
// 헬퍼
// =============================================================

MotionPrimitive BehaviorPlanner::make_translate(const std::vector<double>& q_target, double t_total, TrajectoryProfile profile) {
    MotionPrimitive motion;
    motion.type     = MotionType::TRANSLATE;
    motion.space    = TrajectorySpace::JOINT;
    motion.profile  = profile;
    motion.q_target = q_target;
    motion.t_total  = t_total;
    return motion;
}

void BehaviorPlanner::set_last_q_target(const std::vector<double>& q) {
    last_q_target = q;
    std::lock_guard<std::mutex> lk(ctx.last_q_mutex);
    ctx.last_q_target_snapshot = q;
}

MotionPrimitive BehaviorPlanner::make_drum_hit(double t, int note_num) {
    MotionPrimitive motion;
    motion.type     = MotionType::DRUM;

    DrumEvent Dummy;
    motion.robotic_drum_score.push_back(Dummy);     // rds[0]

    DrumEvent event;
    event.bar = 1;
    event.t = t;
    if (note_num == 0) {
        event.is_kick = true;
    } else  if (note_num == 2 || note_num == 3 || note_num == 6 || note_num == 7) {
        event.note_num_R = note_num;
        event.velocity_R = 5;
    } else {
        event.note_num_L = note_num;
        event.velocity_L = 5;
        if (note_num == 5) event.is_closed_hihat = true;
    }

    motion.robotic_drum_score.push_back(event);     // rds[1]
    return motion;
}

std::string BehaviorPlanner::trim_whitespace(const std::string &str) {
    size_t first = str.find_first_not_of(" \t");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, (last - first + 1));
}

bool BehaviorPlanner::make_drum_event(const std::vector<std::string>& items, double bpm, double last_t, DrumEvent& out) {
    if (items.size() < 8) {
        std::cerr << "[BehaviorPlanner] PLAY: 악보 열 개수 부족 (" << items.size() << ")\n";
        return false;
    }
    try {
        out.bar             = std::stoi(items[0]);
        out.beat            = std::stod(items[1]);
        out.note_num_R      = std::stoi(items[2]);
        out.note_num_L      = std::stoi(items[3]);
        out.velocity_R      = std::stoi(items[4]);
        out.velocity_L      = std::stoi(items[5]);
        out.is_kick         = (std::stoi(items[6]) == 1);
        out.is_closed_hihat = (std::stoi(items[7]) == 1);
    } catch (const std::exception& e) {
        std::cerr << "[BehaviorPlanner] PLAY: 악보 숫자 파싱 실패: " << e.what() << "\n";
        return false;
    }
    out.t = out.beat * 100.0 / bpm + last_t;
    return true;
}

MotionPrimitive BehaviorPlanner::make_drum_play(std::vector<DrumEvent> rds) {
    MotionPrimitive motion;
    motion.type = MotionType::DRUM;
    motion.robotic_drum_score = rds;
    return motion;
}

int BehaviorPlanner::find_motor_id(const std::string& motor_name) const {
    for (const auto &[id, name] : robot.joint_names) {
        if (name == motor_name) return id;
    }
    return -1;
}