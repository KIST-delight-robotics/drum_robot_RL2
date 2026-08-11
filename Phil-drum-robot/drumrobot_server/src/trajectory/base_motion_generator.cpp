#include "trajectory/base_motion_generator.hpp"

BaseMotionGenerator::BaseMotionGenerator() {

}

BaseMotionGenerator::~BaseMotionGenerator() {

}

void BaseMotionGenerator::initialize(const std::map<int, InstrumentCoordinate>& coordinates) {
 
    solver.initialize();
    drum_coordinates = coordinates;
}

BaseMotionPoint BaseMotionGenerator::reset(int note_r, int note_l) {
    right_context = MotionContext{note_r};
    left_context = MotionContext{note_l};

    BaseMotionPoint point;
    note_to_target(note_r, Arm::RIGHT, point.right_position, point.right_wrist);
    note_to_target(note_l, Arm::LEFT, point.left_position, point.left_wrist);

    auto [opt, range] = compute_waist_range(point.right_position, point.left_position, point.right_wrist, point.left_wrist);
    point.waist = opt;

    prev_waist_angle = opt;
    cur_waist_angle = opt;
    prev_t = -1.0;
    cur_q0_min = range[0];
    cur_q0_max = range[1];

    base_end_error = false;

    return point;
}

std::queue<BaseMotionPoint> BaseMotionGenerator::generate_motion(const std::vector<DrumEvent>& rds, int num_point, double dt) {
    std::queue<BaseMotionPoint> out;
 
    if (rds.size() < 2 || num_point <= 0) {
        return out;
    }

    MotionSegment seg_R = get_motion_segment(rds, Arm::RIGHT, right_context);
    MotionSegment seg_L = get_motion_segment(rds, Arm::LEFT, left_context);

    WaistSegment seg_w = get_waist_segment(rds);

    right_context = seg_R.next_context; // context update
    left_context = seg_L.next_context;

    // std::cout << "===== R =====\n";
    // std::cout << "start_time: " << seg_R.start_time << "\n";
    // std::cout << "end_time: " << seg_R.end_time << "\n";
    // std::cout << "t0: " << seg_R.t0 << "\n";
    // std::cout << "t1: " << seg_R.t1 << "\n";

    // std::cout << "===== L =====\n";
    // std::cout << "start_time: " << seg_L.start_time << "\n";
    // std::cout << "end_time: " << seg_L.end_time << "\n";
    // std::cout << "t0: " << seg_L.t0 << "\n";
    // std::cout << "t1: " << seg_L.t1 << "\n";

    for (int i = 0; i < num_point; i++) {
        BaseMotionPoint point;

        double t_R = i * dt + seg_R.t0 - seg_R.start_time;
        double t_L = i * dt + seg_L.t0 - seg_L.start_time;

        double s_R = time_scaling(0.0, seg_R.end_time - seg_R.start_time, t_R);
        double s_L = time_scaling(0.0, seg_L.end_time - seg_L.start_time, t_L);
        
        point.right_position = make_path(seg_R.start_position, seg_R.end_position, s_R);
        point.left_position = make_path(seg_L.start_position, seg_L.end_position, s_L);

        point.right_wrist = s_R * (seg_R.end_wrist_angle - seg_R.start_wrist_angle) + seg_R.start_wrist_angle;
        point.left_wrist = s_L * (seg_L.end_wrist_angle - seg_L.start_wrist_angle) + seg_L.start_wrist_angle;

        double t_w = i * dt + seg_w.t0;
        point.waist = cubic_hermite(seg_w.t0, seg_w.q0, seg_w.v0, seg_w.t1, seg_w.q1, seg_w.v1, t_w);

        out.push(point);
    }

    return out;
}

bool BaseMotionGenerator::get_error() {
    return base_end_error;
}

BaseMotionGenerator::MotionSegment BaseMotionGenerator::get_motion_segment(const std::vector<DrumEvent>& rds, Arm arm, const MotionContext& context) {
    MotionSegment seg;

    seg.t0 = rds[0].t;
    seg.t1 = rds[1].t;

    // 팔에 따라 참조할 note 선택
    auto note_of = [&](int i) {
        int note = (arm == Arm::RIGHT) ? rds[i].note_num_R : rds[i].note_num_L;
        if (note == 5 && !rds[i].is_closed_hihat) note = 9; // 오픈 하이햇 처리
        return note;
    };

    // ===== 타격 감지 =====
    const double e = 0.00001;
    int rds_size = (int)rds.size();

    bool   is_hit   = false;
    double t_hit    = 0.0;
    int    note_hit = 0;

    for (int i = 1; i < rds_size; i++) {
        if (round(10000 * (HIT_DETECTION_THRESHOLD + e)) < round(10000 * (rds[i].t - rds[0].t))) {
            break;
        }

        if (note_of(i) != 0) {
            is_hit   = true;
            t_hit    = rds[i].t;
            note_hit = note_of(i);
            break;
        }
    }

    // ===== 악기 & 시간 =====
    MotionContext next_context;

    if (note_of(0) == 0) {
        if (context.state == State::REST_TO_HIT || context.state == State::HIT_TO_HIT) {
            // 기존의 이동하던 궤적을 이어서 이동
            note_to_target(context.last_instrument, arm,
                           seg.start_position, seg.start_wrist_angle);
            note_to_target(note_hit, arm,
                           seg.end_position, seg.end_wrist_angle);

            seg.start_time = context.last_t;
            seg.end_time   = t_hit;

            next_context = context;
        } else {
            if (is_hit) {
                // 휴식 중 다음 타격 찾음
                note_to_target(context.last_instrument, arm,
                               seg.start_position, seg.start_wrist_angle);
                note_to_target(note_hit, arm,
                               seg.end_position, seg.end_wrist_angle);

                seg.start_time = rds[0].t;
                seg.end_time   = t_hit;

                next_context.state          = State::REST_TO_HIT;
                next_context.last_t         = rds[0].t;
                next_context.last_instrument = context.last_instrument;

            } else {
                // 휴식 중 다음 타격 없음 (현재 위치 유지)
                note_to_target(context.last_instrument, arm,
                               seg.start_position, seg.start_wrist_angle);
                seg.end_position    = seg.start_position;
                seg.end_wrist_angle = seg.start_wrist_angle;

                seg.start_time = rds[0].t;
                seg.end_time   = rds[1].t;

                next_context.state          = State::REST_TO_REST;
                next_context.last_t         = rds[0].t;
                next_context.last_instrument = context.last_instrument;
            }
        }
    } else {
        if (is_hit) {
            // 타격 후 다음 타격 찾음
            note_to_target(note_of(0), arm,
                           seg.start_position, seg.start_wrist_angle);
            note_to_target(note_hit, arm,
                           seg.end_position, seg.end_wrist_angle);

            seg.start_time = rds[0].t;
            seg.end_time   = t_hit;

            next_context.state          = State::HIT_TO_HIT;
            next_context.last_t         = rds[0].t;
            next_context.last_instrument = note_of(0);
        } else {
            // 타격 후 다음 타격 없음 (현재 위치 유지)
            note_to_target(note_of(0), arm,
                           seg.start_position, seg.start_wrist_angle);
            seg.end_position    = seg.start_position;
            seg.end_wrist_angle = seg.start_wrist_angle;

            seg.start_time = rds[0].t;
            seg.end_time   = rds[1].t;

            next_context.state          = State::HIT_TO_REST;
            next_context.last_t         = rds[0].t;
            next_context.last_instrument = note_of(0);
        }
    }

    seg.next_context = next_context;

    return seg;
}

void BaseMotionGenerator::note_to_target(int note_num, Arm arm, std::array<double, 3>& out_position, double& out_wrist_angle_deg) {
    auto it = drum_coordinates.find(note_num);
    if (it == drum_coordinates.end()) {
        std::cerr << "[BaseMotionGenerator] note_to_target: invalid note number "
                  << note_num << "\n";
        out_position = {0.0, 0.0, 0.0};
        out_wrist_angle_deg = 0.0;
        base_end_error = true;  // 악보 오류: 연주 종료
        return;
    }
    const InstrumentCoordinate& coord = it->second;
    if (arm == Arm::RIGHT) { out_position = coord.right_position; out_wrist_angle_deg = coord.right_wrist_angle; }
    else                   { out_position = coord.left_position;  out_wrist_angle_deg = coord.left_wrist_angle; }
}

double BaseMotionGenerator::time_scaling(double ti, double tf, double t) {
    if (tf <= ti) {
        return (t < ti) ? 0.0 : 1.0;   // 0으로 나누기 방지
    }

    double s = (t - ti) / (tf - ti);

    // 구간 밖 클램프
    if (s <= 0.0) return 0.0;
    if (s >= 1.0) return 1.0;

    return s * s * (3.0 - 2.0 * s);   // 3s^2 - 2s^3
}

std::array<double, 3> BaseMotionGenerator::make_path(const std::array<double, 3>& pi, const std::array<double, 3>& pf, double s) {
    std::array<double, 3> ps;

    if (pi == pf) {
        ps = pi;
    } else {
        double h1 = 0.0, h2 = 0.08 + 0.6 * std::abs(pf[2] - pi[2]);     // 조절할 수 있음
        
        std::array<double, 3> pm1;
        std::array<double, 3> pm2;
        for (int i = 0; i < 2; i++) {
            pm1[i] = (2.0/3.0) * pi[i] + (1.0/3.0) * pf[i];
            pm2[i] = (1.0/3.0) * pi[i] + (2.0/3.0) * pf[i];
        }
        pm1[2] = pi[2] + h1;
        pm2[2] = pf[2] + h2;

        // 3차 Bézier curve
        double u = 1.0 - s;
        double b0 = u*u*u;
        double b1 = 3.0*u*u*s;
        double b2 = 3.0*u*s*s;
        double b3 = s*s*s;
        for (int i = 0; i < 3; i++) {
            ps[i] = b0*pi[i] + b1*pm1[i] + b2*pm2[i] + b3*pf[i];
        }
    }

    return ps;
}

BaseMotionGenerator::WaistSegment BaseMotionGenerator::get_waist_segment(const std::vector<DrumEvent>& rds) {
    std::array<double, 4> t_03{};
    std::array<double, 4> q0_opt{};
    std::array<double, 4> q0_min{};
    std::array<double, 4> q0_max{};

    t_03[0] = rds[0].t;
    q0_opt[0] = cur_waist_angle;
    q0_min[0] = cur_q0_min;
    q0_max[0] = cur_q0_max;

    for (int i = 1; i < 4; i++) {
        if ((int)rds.size() > i) {
            t_03[i] = rds[i].t;
            auto [opt, range] = get_waist_angle(rds, i);
            q0_opt[i] = opt;
            q0_min[i] = range[0];
            q0_max[i] = range[1];
        } else {
            t_03[i] = t_03[i - 1] + 1.0;
            q0_opt[i] = q0_opt[i - 1];
            q0_min[i] = q0_min[i - 1];
            q0_max[i] = q0_max[i - 1];
        }
    }

    // 기울기 평균 이동: t0 -> t1
    std::array<double, 3> a0{};
    for (int i = 0; i < 3; i++) {
        a0[i] = (q0_opt[i + 1] - cur_waist_angle) / (t_03[i + 1] - t_03[0]);
    }
    double avg_a0 = (a0[0] + a0[1] + a0[2]) / 3.0;
    double next_waist_angle = cur_waist_angle + avg_a0 * (t_03[1] - t_03[0]);

    if (next_waist_angle <= q0_min[1] || next_waist_angle >= q0_max[1]) {
        next_waist_angle = (q0_min[1] + q0_max[1]) / 2.0;
    }

    // 기울기 평균 이동: t1 -> t2
    std::array<double, 2> a1{};
    for (int i = 0; i < 2; i++) {
        a1[i] = (q0_opt[i + 2] - next_waist_angle) / (t_03[i + 2] - t_03[1]);
    }
    double avg_a1 = (a1[0] + a1[1]) / 2.0;
    double next_next_waist_angle = next_waist_angle + avg_a1 * (t_03[2] - t_03[1]);

    if (next_next_waist_angle <= q0_min[2] || next_next_waist_angle >= q0_max[2]) {
        next_next_waist_angle = (q0_min[2] + q0_max[2]) / 2.0;
    }

    // 기울기 계산
    std::array<double, 4> q = {prev_waist_angle, cur_waist_angle, next_waist_angle, next_next_waist_angle};
    std::array<double, 4> t = {prev_t, t_03[0], t_03[1], t_03[2]};

    std::array<double, 2> m = compute_slopes(q, t);

    // 반환
    WaistSegment seg;

    seg.t0 = t_03[0];
    seg.t1 = t_03[1];
    
    seg.q0 = cur_waist_angle;
    seg.q1 = next_waist_angle;

    seg.v0 = m[0];
    seg.v1 = m[1];

    prev_t = t_03[0];
    prev_waist_angle = cur_waist_angle;
    cur_waist_angle = next_waist_angle;
    cur_q0_min = q0_min[1];
    cur_q0_max = q0_max[1];

    return seg;
}

std::pair<double, std::array<double, 2>> BaseMotionGenerator::get_waist_angle(const std::vector<DrumEvent>& rds, int idx) {
    // idx 번째 허리 최적값, 범위 구하기

    MotionContext tmp_context_R = right_context;
    MotionContext tmp_context_L = left_context;

    for (int i = 0; i < (int)rds.size() - 1; i++) {
        MotionSegment seg_R = get_motion_segment(std::vector<DrumEvent>(rds.begin() + i, rds.end()), Arm::RIGHT, tmp_context_R);
        MotionSegment seg_L = get_motion_segment(std::vector<DrumEvent>(rds.begin() + i, rds.end()), Arm::LEFT, tmp_context_L);

        if (i + 1 == idx) {
            double t_R = seg_R.t1 - seg_R.start_time;
            double t_L = seg_L.t1 - seg_L.start_time;

            double s_R = time_scaling(0.0, seg_R.end_time - seg_R.start_time, t_R);
            double s_L = time_scaling(0.0, seg_L.end_time - seg_L.start_time, t_L);
            
            std::array<double, 3> right_position = make_path(seg_R.start_position, seg_R.end_position, s_R);
            std::array<double, 3> left_position = make_path(seg_L.start_position, seg_L.end_position, s_L);

            double right_wrist = s_R * (seg_R.end_wrist_angle - seg_R.start_wrist_angle) + seg_R.start_wrist_angle;
            double left_wrist = s_L * (seg_L.end_wrist_angle - seg_L.start_wrist_angle) + seg_L.start_wrist_angle;

            return compute_waist_range(right_position, left_position, right_wrist, left_wrist); 
        }

        tmp_context_R = seg_R.next_context;
        tmp_context_L = seg_L.next_context;
    }

    base_end_error = true;  // 인덱싱 오류: 연주 종료
    std::array<double, 2> err{};
    return {0.0, err};
}

std::pair<double, std::array<double, 2>> BaseMotionGenerator::compute_waist_range(std::array<double, 3> pR, std::array<double, 3> pL, double the7, double the8) {
    std::vector<std::array<double, 9>> q_vec;
    int num_sol = 0;

    for (int i = 0; i < 1801; i++) {
        double the0 = -0.5 * M_PI + M_PI / 1800.0 * i;  // 범위 : -90deg ~ 90deg

        KinematicsSolver::IKResult result = solver.solve_ik(pR, pL, the0, the7, the8, false);

        if (result.success) {
            q_vec.push_back(result.q);
            num_sol++;
        }
    }

    double w0 = 2.0, w1 = 1.0;
    double min_cost = w0 + w1;
    int min_idx = 0;
    double w, cost = 0.0;

    if (num_sol == 0) {
        std::cerr << "[BaseMotionGenerator] 허리 -90~90deg 전 범위에서 IK 해 없음\n";
        base_end_error = true;  // IK 오류: 연주 종료
        std::array<double, 2> err{};
        return {0.0, err};
    } else {
        std::array<double, 2> range = {q_vec[0][0], q_vec[num_sol - 1][0]};
        w = 2.0 * M_PI / std::abs(range[1] - range[0]);

        for (int i = 0; i < num_sol; i++) {
            // 양 팔의 관절각 합이 180도에 가까운 값 선택 + solution set 중 가운데 값 선택
            cost = w0 * cos(q_vec[i][1] + q_vec[i][2]) + w1 * cos(w * std::abs(q_vec[i][0] - range[0]));

            if (cost < min_cost) {
                min_cost = cost;
                min_idx = i;
            }
        }

        return {q_vec[min_idx][0], range};
    }
}

std::array<double, 2> BaseMotionGenerator::compute_slopes(const std::array<double, 4> &q, const std::array<double, 4> &t) {
    // Monotone Cubic Interpolation 을 위한 기울기 계산
    std::array<double, 2> m{};
    std::array<double, 3> a{};

    for (int i = 0; i < 3; i++) {
        a[i] = (q[i + 1] - q[i]) / (t[i + 1] - t[i]);
    }
    double m1 = 0.5 * (a[0] + a[1]);
    double m2 = 0.5 * (a[1] + a[2]);

    double alph, bet;
    if (q[1] == q[2]) {
        m1 = 0;
        m2 = 0;
    } else {
        if((q[0] == q[1]) || (a[0] * a[1] < 0)){
            m1 = 0;
        } else if((q[2] == q[3]) || (a[1] * a[2] < 0)){
            m2 = 0;
        }
        alph = m1 / (q[2] - q[1]);
        bet = m2 / (q[2] - q[1]);

        double e = std::sqrt(std::pow(alph, 2) + std::pow(bet, 2));
        if (e > 3.0) {
            m1 = (3 * m1) / e;
            m2 = (3 * m2) / e;
        }
    }
    
    m[0] = m1;
    m[1] = m2;
    
    return m;
}

double BaseMotionGenerator::cubic_hermite(double ta, double qa, double va, double tb, double qb, double vb, double t) {
    // [ta,tb] 구간을 q(ta)=qa, q'(ta)=va, q(tb)=qb, q'(tb)=vb 로 보간하는 3차 다항식.
    double T = tb - ta;
    if (T == 0.0) {
        return qa;
    }
    double tau = (t - ta) / T;      // 0 ~ 1

    double h00 =  2.0*tau*tau*tau - 3.0*tau*tau + 1.0;
    double h10 =      tau*tau*tau - 2.0*tau*tau + tau;
    double h01 = -2.0*tau*tau*tau + 3.0*tau*tau;
    double h11 =      tau*tau*tau -     tau*tau;

    return h00*qa + h10*T*va + h01*qb + h11*T*vb;
}