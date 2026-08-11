#include "trajectory/trajectory_generator.hpp"

#include <fstream>
#include <sstream>

TrajectoryGenerator::TrajectoryGenerator(AppContext& ctxRef, ControlQueue &controlQueueRef)
    : ctx(ctxRef), control_queue(controlQueueRef),
      play_motion_generator(ctxRef), trajectory_log("trajectory") {
    
    std::vector<std::string> header = {
        "joint 0", "joint 1", "joint 2", "joint 3", "joint 4",
        "joint 5", "joint 6", "joint 7", "joint 8",
        "joint 9", "joint 10", "joint 11", "joint 12"
    };
    trajectory_log.set_header(header);
}

TrajectoryGenerator::~TrajectoryGenerator() {

}

void TrajectoryGenerator::initialize(const std::map<std::string, std::vector<double>>& pose) {
    solver.initialize();
    play_motion_generator.initialize();

    update_last_q(pose.at("init"));
    ready_pose = pose.at("ready");
}

void TrajectoryGenerator::generate_trajectory(const MotionPrimitive& motion) {
    switch (motion.type) {
    case MotionType::STANDBY:
        generate_standby_trajectory();  // 키 제거하기 전 현재 위치 유지 (고정)
        break;
    case MotionType::TRANSLATE:
        if (motion.space == TrajectorySpace::JOINT) {
            generate_joint_space_trajectory(motion);
        } else if (motion.space == TrajectorySpace::TASK) {
            generate_task_space_trajectory(motion);
        } else {
            std::cerr << "[TrajectoryGenerator] Unknown motion trajectory space\n";
        }
        break;
    case MotionType::DRUM:
        if (motion.flag == PlayFlag::START) {
            generate_play_start_trajectory(motion);
        } else if (motion.flag == PlayFlag::END) {
            generate_play_end_trajectory();
        } else {
            generate_play_trajectory(motion);
        }
        break;
    case MotionType::IDLE:
        generate_idle_trajectory();
        break;
    case MotionType::REPLAY:
        generate_replay_trajectory(motion);
        break;
    default:
        std::cerr << "[TrajectoryGenerator] Unknown motion type\n";
        break;
    }
}

void TrajectoryGenerator::generate_standby_trajectory() {
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = get_modes();
    double t_total = 1.0;
    int num_point = static_cast<int>(t_total / ROBOT::DT_SECOND);

    std::vector<double> q0(last_q.begin(), last_q.end());
    std::vector<double> q1(last_q.begin(), last_q.end());

    for (int k = 1; k <= num_point; k++) {
        auto [q, qd] = sample_cosine(q0, q1, num_point, k);

        ControlSetPoint set_point;
        std::copy(q.begin(),  q.end(),  set_point.q.begin());
        std::copy(qd.begin(), qd.end(), set_point.qd.begin());
        set_point.mode = modes;
        control_queue.push(set_point);

        trajectory_log.record(set_point.q);
    }
}

void TrajectoryGenerator::generate_joint_space_trajectory(const MotionPrimitive& motion) {
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = get_modes();
    int num_point = static_cast<int>(motion.t_total / ROBOT::DT_SECOND);

    std::vector<double> q0(last_q.begin(), last_q.end());
    std::vector<double> q1 = motion.q_target;

    for (int k = 1; k <= num_point; k++) {
        auto [q, qd] = sample(q0, q1, num_point, k, motion.profile);

        ControlSetPoint set_point;
        std::copy(q.begin(),  q.end(),  set_point.q.begin());
        std::copy(qd.begin(), qd.end(), set_point.qd.begin());
        set_point.mode = modes;
        control_queue.push(set_point);

        trajectory_log.record(set_point.q);
    }

    update_last_q(q1);
}

void TrajectoryGenerator::generate_task_space_trajectory(const MotionPrimitive& motion) {
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = get_modes();
    int num_point = static_cast<int>(motion.t_total / ROBOT::DT_SECOND);

    std::vector<double> q0(last_q.begin(), last_q.end());
    std::vector<double> q1 = motion.q_target;

    std::vector<double> p0 = {
        last_p_R[0], last_p_R[1], last_p_R[2],
        last_p_L[0], last_p_L[1], last_p_L[2]
    };

    std::vector<double> p1 = {
        motion.p_target_R[0], motion.p_target_R[1], motion.p_target_R[2],
        motion.p_target_L[0], motion.p_target_L[1], motion.p_target_L[2]
    };

    // 속도 계산을 위한 이전 관절각
    std::array<double, 9> prev_q;
    std::copy(last_q.begin(), last_q.begin() + 9, prev_q.begin());

    std::queue<ControlSetPoint> buffer;

    for (int k = 1; k <= num_point; k++) {
        auto [q, qd] = sample(q0, q1, num_point, k, motion.profile);
        auto [p, pd] = sample(p0, p1, num_point, k, motion.profile);

        std::array<double, 3> pR = {p[0], p[1], p[2]};
        std::array<double, 3> pL = {p[3], p[4], p[5]};
        double theta0 = q[0];
        double theta7 = q[7];
        double theta8 = q[8];
        KinematicsSolver::IKResult result = solver.solve_ik(pR, pL, theta0, theta7, theta8, true);

        if (!result.success) {
            std::cerr << "[TrajectoryGenerator] Failed to solve inverse kinematics\n";
            return;
        }

        // 13차원 set_point 구성: IK 결과(0~8) + 관절 보간값(9~12)
        ControlSetPoint set_point;
        for (int i = 0; i < 9; i++) {
            set_point.q[i] = result.q[i];   // 관절 0~8 (팔)
            set_point.qd[i] = (result.q[i] - prev_q[i]) / ROBOT::DT_SECOND;
        }
        for (int i = 9; i < ROBOT::NUM_JOINT; i++) {
            set_point.q[i] = q[i];          // 관절 9~12 (페달, 머리)
            set_point.qd[i] = qd[i]; 
        }
        set_point.mode = modes;
        buffer.push(set_point);

        prev_q = result.q;
    }

    // IK 오류가 없으면 적재
    while (!buffer.empty()) {
        ControlSetPoint sp = buffer.front();
        buffer.pop();
        control_queue.push(sp);

        trajectory_log.record(sp.q);
    }

    update_last_q(p1, q1);
}

void TrajectoryGenerator::generate_play_start_trajectory(const MotionPrimitive& motion) {
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = get_modes();
    double t_total = 4.0;
    int num_point = static_cast<int>(t_total / ROBOT::DT_SECOND);

    std::array<double, ROBOT::NUM_JOINT> q_target;
    if (play_motion_generator.reset(q_target, motion.init_note_r, motion.init_note_l)) {
        std::vector<double> q0(last_q.begin(), last_q.end());
        std::vector<double> q1(q_target.begin(), q_target.end());

        for (int k = 1; k <= num_point; k++) {
            auto [q, qd] = sample(q0, q1, num_point, k, TrajectoryProfile::COSINE);

            ControlSetPoint set_point;
            std::copy(q.begin(),  q.end(),  set_point.q.begin());
            std::copy(qd.begin(), qd.end(), set_point.qd.begin());
            set_point.mode = modes;
            control_queue.push(set_point);

            trajectory_log.record(set_point.q);
        }

        update_last_q(q1);

        first_point = true; // 음악 재생을 위함
    } else {
        ctx.play_abort = true;
    }
}

void TrajectoryGenerator::generate_play_end_trajectory() {
    // play 후 레디 자세로 이동
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = get_modes();
    double t_total = 4.0;
    int num_point = static_cast<int>(t_total / ROBOT::DT_SECOND);

    std::vector<double> q0(last_q.begin(), last_q.end());
    std::vector<double> q1 = ready_pose;

    for (int k = 1; k <= num_point; k++) {
        auto [q, qd] = sample(q0, q1, num_point, k, TrajectoryProfile::COSINE);

        ControlSetPoint set_point;
        std::copy(q.begin(),  q.end(),  set_point.q.begin());
        std::copy(qd.begin(), qd.end(), set_point.qd.begin());
        set_point.mode = modes;
        control_queue.push(set_point);

        trajectory_log.record(set_point.q);
    }

    update_last_q(q1);
}

void TrajectoryGenerator::generate_play_trajectory(const MotionPrimitive& motion) {
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = get_modes(true);

    std::queue<std::array<double, ROBOT::NUM_JOINT>> play_motion = play_motion_generator.generate_motion(motion.robotic_drum_score);

    // 속도 계산을 위한 이전 관절각
    std::array<double, ROBOT::NUM_JOINT> prev_q = last_q;

    if (play_motion.empty()) {
        ctx.play_abort = true;
        return;
    }

    while (!play_motion.empty()) {
        std::array<double, ROBOT::NUM_JOINT> q = play_motion.front();
        play_motion.pop();

        ControlSetPoint set_point;
        for (int i = 0; i < ROBOT::NUM_JOINT; i++) {
            set_point.q[i] = q[i];
            set_point.qd[i] = (q[i] - prev_q[i]) / ROBOT::DT_SECOND;    // TODO: 수치 미분 스파이크 무서우면 필터 넣기
        }

        // 싱크 맞춰서 음악 재생
        if (first_point) {
            first_point = false;
            set_point.audio_start = true;
        }
        
        set_point.mode = modes;
        control_queue.push(set_point);

        prev_q = q;

        trajectory_log.record(set_point.q);
    }

    update_last_q(prev_q);
}

void TrajectoryGenerator::generate_idle_trajectory() {
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = get_modes();
    double t_total = 1.0;
    int num_point = static_cast<int>(t_total / ROBOT::DT_SECOND);

    std::vector<double> q0(last_q.begin(), last_q.end());
    std::vector<double> q1(last_q.begin(), last_q.end());    // TODO: 미세한 움직임 구현

    for (int k = 1; k <= num_point; k++) {
        auto [q, qd] = sample_cosine(q0, q1, num_point, k);

        ControlSetPoint set_point;
        std::copy(q.begin(),  q.end(),  set_point.q.begin());
        std::copy(qd.begin(), qd.end(), set_point.qd.begin());
        set_point.mode = modes;
        control_queue.push(set_point);

        trajectory_log.record(set_point.q);
    }

    update_last_q(q1);
}

// ============================================================================
// Stage A: 미리 뽑은 관절궤적 CSV를 오픈루프로 재생 (실기 trajectory_log 양식과 동일)
//   CSV 포맷: 헤더 1줄 "t,joint 0,...,joint 12" + 각 행 "t,q0..q12"
//             (관절각 rad, 실기 모터 id 순서). → 서버 trajectory_log 출력 양식 그대로.
//             sim(play1.py) export 도 이 양식으로 뽑으므로 sim궤적/서버로그 둘 다 재생 가능.
//   - 규약 변환(sim→실기, 손목 등)은 exporter(play1.py)에서 끝냄 → 여기선 무변환 스트리밍.
//   - col0=t, col1..9 = 관절 0..8(팔). 페달/머리(9~12)는 무시하고 현재값(last_q) 유지.
//   - CSV t열로 5ms(200Hz) 그리드에 선형보간 재타이밍.
//   - 팔(0~6)=POS, 손목(7,8)=CSP, 페달(9,10)=CSP(현재값 유지), 머리(11,12)=NONE.
//   - 시작점 급변 방지용으로 현재자세->CSV 첫자세 2s 코사인 램프 선행.
//   ⚠ 미검증 실기 코드: 반드시 저게인·드럼 비접촉·E-stop 대기 상태에서 먼저 확인.
// ============================================================================
void TrajectoryGenerator::generate_replay_trajectory(const MotionPrimitive& motion) {
    std::ifstream file(motion.replay_csv);
    if (!file.is_open()) {
        std::cerr << "[Replay] CSV 열기 실패: " << motion.replay_csv << "\n";
        return;
    }

    // 한 줄을 콤마로 분리 + trim
    auto split_csv = [](const std::string& line) {
        std::vector<std::string> out;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            size_t a = tok.find_first_not_of(" \t\r\n");
            size_t b = tok.find_last_not_of(" \t\r\n");
            out.push_back(a == std::string::npos ? "" : tok.substr(a, b - a + 1));
        }
        return out;
    };

    // 헤더 1줄 스킵 (t,joint 0,...,joint 12)
    std::string header_line;
    if (!std::getline(file, header_line)) {
        std::cerr << "[Replay] 빈 CSV\n";
        return;
    }

    // 데이터: col0=t, col1..9 = 관절 0..8(팔). (페달/머리 열은 있어도 무시)
    std::vector<double> times;
    std::vector<std::array<double, 9>> rows;   // 실기 id 순서 팔 0~8, rad
    std::string line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::vector<std::string> f = split_csv(line);
        if (f.size() < 10) continue;   // t + 최소 팔 9관절
        times.push_back(std::stod(f[0]));
        std::array<double, 9> q;
        for (int id = 0; id < 9; id++) q[id] = std::stod(f[id + 1]);
        rows.push_back(q);
    }
    file.close();

    if (rows.size() < 2) {
        std::cerr << "[Replay] 데이터 행 부족 (" << rows.size() << ")\n";
        return;
    }
    std::cerr << "[Replay] " << rows.size() << " 행 로드 ("
              << times.front() << "s ~ " << times.back() << "s)\n";

    // 모드: 팔=POS, 손목=CSP, 페달=CSP(현재값 유지), 머리=NONE
    std::array<ControlMode, ROBOT::NUM_JOINT> modes = {
        ControlMode::POS, ControlMode::POS, ControlMode::POS,
        ControlMode::POS, ControlMode::POS, ControlMode::POS, ControlMode::POS,
        ControlMode::CSP, ControlMode::CSP,     // 손목 7,8
        ControlMode::CSP, ControlMode::CSP,     // 페달 9,10
        ControlMode::NONE, ControlMode::NONE    // 머리 11,12
    };

    // CSV 첫 자세 -> 13차원 (팔은 CSV, 페달/머리는 현재 last_q 유지)
    std::vector<double> q_first(last_q.begin(), last_q.end());
    for (int id = 0; id < 9; id++) q_first[id] = rows.front()[id];

    // (1) 현재자세 -> CSV 첫자세 코사인 램프 (시작점 급변/POS_DIFF_LIMIT 방지)
    {
        std::vector<double> q0(last_q.begin(), last_q.end());
        int ramp_n = static_cast<int>(2.0 / ROBOT::DT_SECOND);
        for (int k = 1; k <= ramp_n; k++) {
            auto [q, qd] = sample_cosine(q0, q_first, ramp_n, k);
            ControlSetPoint sp;
            std::copy(q.begin(),  q.end(),  sp.q.begin());
            std::copy(qd.begin(), qd.end(), sp.qd.begin());
            sp.mode = modes;
            control_queue.push(sp);
            trajectory_log.record(sp.q);
        }
    }

    // (2) 본 재생: CSV t열 기준 5ms 그리드 선형보간 스트리밍
    std::array<double, ROBOT::NUM_JOINT> prev_q;
    std::copy(q_first.begin(), q_first.end(), prev_q.begin());

    const double dt = ROBOT::DT_SECOND;
    const double T0 = times.front();
    const double T1 = times.back();
    int num_point = static_cast<int>((T1 - T0) / dt);
    size_t seg = 0;

    for (int k = 1; k <= num_point; k++) {
        double tq = T0 + k * dt;
        while (seg + 1 < times.size() - 1 && times[seg + 1] < tq) seg++;
        double t_a = times[seg];
        double t_b = times[seg + 1];
        double alpha = (t_b > t_a) ? (tq - t_a) / (t_b - t_a) : 0.0;
        if (alpha < 0.0) alpha = 0.0;
        if (alpha > 1.0) alpha = 1.0;

        ControlSetPoint sp;
        for (int id = 0; id < 9; id++) {   // 팔 0~8: CSV 보간
            double v = rows[seg][id] + alpha * (rows[seg + 1][id] - rows[seg][id]);
            sp.q[id]  = v;
            sp.qd[id] = (v - prev_q[id]) / dt;
        }
        for (int id = 9; id < ROBOT::NUM_JOINT; id++) {   // 페달/머리: 현재값 유지
            sp.q[id]  = last_q[id];
            sp.qd[id] = 0.0;
        }
        sp.mode = modes;
        control_queue.push(sp);
        trajectory_log.record(sp.q);

        std::copy(sp.q.begin(), sp.q.end(), prev_q.begin());
    }

    std::vector<double> q_end(prev_q.begin(), prev_q.end());
    update_last_q(q_end);
    std::cerr << "[Replay] 완료: " << num_point << " 포인트 스트리밍\n";
}

std::array<ControlMode, ROBOT::NUM_JOINT> TrajectoryGenerator::get_modes(bool is_play) {
    if (is_play) {
        wrist_control_mode = ControlMode::CST;
    } else {
        wrist_control_mode = ControlMode::CSP;
    }

    std::array<ControlMode, ROBOT::NUM_JOINT> modes = {
        tmotor_control_mode,
        tmotor_control_mode,
        tmotor_control_mode,
        tmotor_control_mode,
        tmotor_control_mode,
        tmotor_control_mode,
        tmotor_control_mode,
        wrist_control_mode,
        wrist_control_mode,
        pedal_control_mode,
        pedal_control_mode,
        ControlMode::NONE,
        ControlMode::NONE
    };

    return modes;
}

std::pair<std::vector<double>, std::vector<double>> TrajectoryGenerator::sample(
    const std::vector<double>& q0,
    const std::vector<double>& q1,
    int n,
    int k,
    TrajectoryProfile profile
) {
    std::pair<std::vector<double>, std::vector<double>> result;

    switch (profile) {
    case TrajectoryProfile::TRAPEZOIDAL:
        result = sample_trapezoidal(q0, q1, n, k);
        break;
    case TrajectoryProfile::CUBIC:
        result = sample_cubic(q0, q1, n, k);
        break;
    case TrajectoryProfile::QUINTIC:
        result = sample_quintic(q0, q1, n, k);
        break;
    case TrajectoryProfile::COSINE:
        result = sample_cosine(q0, q1, n, k);
        break;
    }

    return result;
}

std::pair<std::vector<double>, std::vector<double>> TrajectoryGenerator::sample_trapezoidal(
    const std::vector<double>& q0,
    const std::vector<double>& q1,
    int n,
    int k
) {
    // 사다리꼴 속도 프로파일: 가속(1/4) - 등속(1/2) - 감속(1/4)
    // 정규화 시간 s = k / n  (구간 [0, 1))
    // 정규화 속도 = 위치 미분 / t_total  (qd = ds/dt · (q1 - q0), t_total = n·dt)

    int dim = q0.size();
    std::vector<double> q(dim, 0.0);
    std::vector<double> qd(dim, 0.0);
 
    double s = static_cast<double>(k) / static_cast<double>(n);   // 정규화 시간 [0, 1)
    double t_total = n * ROBOT::DT_SECOND;                                      // 실제 전체 시간 [s]
 
    double s_pos;   // 정규화 위치 [0, 1]
    double s_vel;   // 정규화 속도 ds/d(s_time)
 
    const double t_acc = 0.25;      // 가속 구간 비율
    const double v_max = 4.0 / 3.0; // 최대 정규화 속도 (등속 구간 속도)
 
    if (s < t_acc) {
        // 가속 구간
        s_pos = 0.5 * v_max * (s * s) / t_acc;
        s_vel = v_max * (s / t_acc);
    } else if (s < 1.0 - t_acc) {
        // 등속 구간
        s_pos = 0.5 * v_max * t_acc + v_max * (s - t_acc);
        s_vel = v_max;
    } else {
        // 감속 구간
        double tau = 1.0 - s;
        s_pos = 1.0 - 0.5 * v_max * (tau * tau) / t_acc;
        s_vel = v_max * (tau / t_acc);
    }
    s_pos = std::clamp(s_pos, 0.0, 1.0);
 
    for (int i = 0; i < dim; i++) {
        double dq = q1[i] - q0[i];
        q[i]  = q0[i] + s_pos * dq;
        qd[i] = (s_vel / t_total) * dq;
    }
 
    return {q, qd};
}

std::pair<std::vector<double>, std::vector<double>> TrajectoryGenerator::sample_cubic(
    const std::vector<double>& q0,
    const std::vector<double>& q1,
    int n,
    int k
) {
    // 3차 다항식 보간: 양 끝점에서 속도 0
    // 정규화 시간 s = k / n  (구간 [0, 1))
    // 정규화 속도 = 위치 미분 / t_total  (qd = ds/dt · (q1 - q0), t_total = n·dt)
 
    int dim = q0.size();
    std::vector<double> q(dim, 0.0);
    std::vector<double> qd(dim, 0.0);
 
    double s = static_cast<double>(k) / static_cast<double>(n);
    double t_total = n * ROBOT::DT_SECOND;
 
    double s_pos = 3.0 * s * s - 2.0 * s * s * s;
    double s_vel = 6.0 * s * (1.0 - s);
 
    for (int i = 0; i < dim; i++) {
        double dq = q1[i] - q0[i];
        q[i]  = q0[i] + s_pos * dq;
        qd[i] = (s_vel / t_total) * dq;
    }
 
    return {q, qd};
}

std::pair<std::vector<double>, std::vector<double>> TrajectoryGenerator::sample_quintic(
    const std::vector<double>& q0,
    const std::vector<double>& q1,
    int n,
    int k
) {
    // 5차 다항식 보간: 양 끝점에서 속도와 가속도 모두 0
    // 정규화 시간 s = k / n  (구간 [0, 1))
    // 정규화 속도 = 위치 미분 / t_total  (qd = ds/dt · (q1 - q0), t_total = n·dt)

    int dim = q0.size();
    std::vector<double> q(dim, 0.0);
    std::vector<double> qd(dim, 0.0);
 
    double s = static_cast<double>(k) / static_cast<double>(n);
    double t_total = n * ROBOT::DT_SECOND;
 
    double s2 = s * s;
    double s3 = s2 * s;
    double s4 = s3 * s;
    double s5 = s4 * s;
 
    double s_pos = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
    double s_vel = 30.0 * s2 - 60.0 * s3 + 30.0 * s4;
 
    for (int i = 0; i < dim; i++) {
        double dq = q1[i] - q0[i];
        q[i]  = q0[i] + s_pos * dq;
        qd[i] = (s_vel / t_total) * dq;
    }
 
    return {q, qd};
}

std::pair<std::vector<double>, std::vector<double>> TrajectoryGenerator::sample_cosine(
    const std::vector<double>& q0,
    const std::vector<double>& q1,
    int n,
    int k
) {
    // 사인(코사인) 보간: s(t) = (1 - cos(πt)) / 2
    // 양 끝점에서 속도 0, 매끄러운 가속/감속
    // 정규화 시간 s = k / n  (구간 [0, 1))
    // 정규화 속도 = 위치 미분 / t_total  (qd = ds/dt · (q1 - q0), t_total = n·dt)
 
    int dim = q0.size();
    std::vector<double> q(dim, 0.0);
    std::vector<double> qd(dim, 0.0);
 
    double s = static_cast<double>(k) / static_cast<double>(n);
    double t_total = n * ROBOT::DT_SECOND;
 
    double s_pos = 0.5 * (1.0 - std::cos(M_PI * s));
    double s_vel = 0.5 * M_PI * std::sin(M_PI * s);
 
    for (int i = 0; i < dim; i++) {
        double dq = q1[i] - q0[i];
        q[i]  = q0[i] + s_pos * dq;
        qd[i] = (s_vel / t_total) * dq;
    }
 
    return {q, qd};
}

void TrajectoryGenerator::update_last_q(const std::vector<double>& q) {
    std::copy(q.begin(), q.end(), last_q.begin());
    last_qd.fill(0.0);

    std::array<double, 9> q_in;
    std::copy(last_q.begin(), last_q.begin() + 9, q_in.begin());
    KinematicsSolver::FKResult result = solver.solve_fk(q_in);

    if (!result.success) {
        std::cerr << "[TrajectoryGenerator] Failed to solve forward kinematics\n";
        return;
    }
    
    last_p_R = result.pR;
    last_p_L = result.pL;
}

void TrajectoryGenerator::update_last_q(const std::array<double, ROBOT::NUM_JOINT>& q) {
    last_q = q;
    last_qd.fill(0.0);

    std::array<double, 9> q_in;
    std::copy(last_q.begin(), last_q.begin() + 9, q_in.begin());
    KinematicsSolver::FKResult result = solver.solve_fk(q_in);

    if (!result.success) {
        std::cerr << "[TrajectoryGenerator] Failed to solve forward kinematics\n";
        return;
    }
    
    last_p_R = result.pR;
    last_p_L = result.pL;
}

void TrajectoryGenerator::update_last_q(const std::vector<double>& p, const std::vector<double>& q) {
    std::array<double, 3> pR = {p[0], p[1], p[2]};
    std::array<double, 3> pL = {p[3], p[4], p[5]};
    double theta0 = q[0];
    double theta7 = q[7];
    double theta8 = q[8];
    KinematicsSolver::IKResult result = solver.solve_ik(pR, pL, theta0, theta7, theta8, true);

    if (!result.success) {
        std::cerr << "[TrajectoryGenerator] Failed to solve inverse kinematics\n";
        return;
    }

    // 13차원 last_q 구성: IK 결과(0~8) + 관절 보간값(9~12)
    for (int i = 0; i < 9; i++) {
        last_q[i] = result.q[i];
    }
    for (int i = 9; i < ROBOT::NUM_JOINT; i++) {
        last_q[i] = q[i];
    }
    last_qd.fill(0.0);

    last_p_R = pR;
    last_p_L = pL;
}