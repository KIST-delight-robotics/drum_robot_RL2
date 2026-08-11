#include "trajectory/play_motion_generator.hpp"

PlayMotionGenerator::PlayMotionGenerator(AppContext &ctxRef)
    : ctx(ctxRef) {
}

PlayMotionGenerator::~PlayMotionGenerator() {

}

void PlayMotionGenerator::initialize() {
    using json = nlohmann::json;
 
    solver.initialize();

    const std::string config_path = "drumrobot_server/config/drum_coordinate.json";
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        std::cerr << "[PlayMotionGenerator] Failed to open config file: "
                  << config_path << "\n";
        return;
    }

    json root;
    try {
        ifs >> root;
    } catch (const json::parse_error& e) {
        std::cerr << "[PlayMotionGenerator] JSON parse error in "
                  << config_path << ": " << e.what() << "\n";
        return;
    }

    drum_coordinates.clear();

    for (const auto& inst : root.at("instruments")) {
        InstrumentCoordinate coord;

        std::string name = inst.at("name");
        int id = instrument_name_to_id.at(name);

        const auto& right = inst.at("right");
        const auto& left  = inst.at("left");

        auto right_pos = right.at("position");
        auto left_pos  = left.at("position");

        coord.right_position = {
            right_pos.at(0).get<double>(),
            right_pos.at(1).get<double>(),
            right_pos.at(2).get<double>()
        };
        coord.right_wrist_angle = right.at("wrist_angle_deg").get<double>() * M_PI / 180.0;

        coord.left_position = {
            left_pos.at(0).get<double>(),
            left_pos.at(1).get<double>(),
            left_pos.at(2).get<double>()
        };
        coord.left_wrist_angle = left.at("wrist_angle_deg").get<double>() * M_PI / 180.0;

        drum_coordinates[id] = coord;
    }

    std::cout << "[PlayMotionGenerator] Loaded " << drum_coordinates.size()
              << " drum coordinates from " << config_path << "\n";

    base_motion_generator.initialize(drum_coordinates);
    head_motion_generator.initialize(drum_coordinates);
}

bool PlayMotionGenerator::reset(std::array<double, ROBOT::NUM_JOINT>& q, int note_r, int note_l) {
    BaseMotionPoint b = base_motion_generator.reset(note_r, note_l);
    HeadMotionPoint h = head_motion_generator.reset(note_r);
    PedalMotionPoint p = pedal_motion_generator.reset();
    StateMotionPoint s = state_motion_generator.reset();

    std::array<double, 3> pR = b.right_position;
    std::array<double, 3> pL = b.left_position;
    double theta0 = b.waist;
    double theta7 = b.right_wrist;
    double theta8 = b.left_wrist;
    KinematicsSolver::IKResult result = solver.solve_ik(pR, pL, theta0, theta7, theta8, true);

    if (!result.success) {
        std::cerr << "[PlayMotionGenerator] RESET: Failed to solve inverse kinematics\n";
        return false;
    }

    for (int i = 0; i < 9; i++) {
        q[i] = result.q[i];   // 관절 0~8 (팔)
    }

    q[4] += s.right_elbow;
    q[6] += s.left_elbow;

    q[7] += s.right_wrist;
    q[8] += s.left_wrist;

    q[9] = p.right;
    q[10] = p.left;

    q[11] = h.yaw - q[0];
    q[12] = h.pitch;

    return true;
}

std::queue<std::array<double, ROBOT::NUM_JOINT>> PlayMotionGenerator::generate_motion(const std::vector<DrumEvent>& rds) {    
    if (rds.size() < 2) {
        std::cerr << "[PlayMotionGenerator] generate_motion: 드럼 이벤트가 부족합니다 (size="
                  << rds.size() << ", 최소 2개 필요). 해당 구간 생성을 건너뜁니다.\n";
        std::queue<std::array<double, ROBOT::NUM_JOINT>> empty_queue;
        return empty_queue;
    }

    // std::cout << "===== rds =====\n";
    // for (int i = 0; i < (int)rds.size(); i++) {
    //     std::cout << "[" << i << "] t: " << rds[i].t
    //               << "  note_R: " << rds[i].note_num_R
    //               << "  note_L: " << rds[i].note_num_L
    //               << "  vel_R: " << rds[i].velocity_R
    //               << "  vel_L: " << rds[i].velocity_L << "\n";
    // }

    std::queue<std::array<double, ROBOT::NUM_JOINT>> q_queue;
    auto [n, dt] = get_num_point(rds[0].t, rds[1].t);

    std::queue<BaseMotionPoint> base_motion = base_motion_generator.generate_motion(rds, n, dt);
    std::queue<HeadMotionPoint> head_motion = head_motion_generator.generate_motion(rds, n);
    std::queue<PedalMotionPoint> pedal_motion = pedal_motion_generator.generate_motion(rds, n, dt);
    std::queue<StateMotionPoint> state_motion = state_motion_generator.generate_motion(rds, n, dt);

    if (base_motion_generator.get_error() || state_motion_generator.get_error()) {
        std::queue<std::array<double, ROBOT::NUM_JOINT>> empty_queue;
        return empty_queue;
    }

    for (int i = 0; i < n; i++) {
        std::array<double, ROBOT::NUM_JOINT> q;

        BaseMotionPoint b = base_motion.front();
        base_motion.pop();

        HeadMotionPoint h = head_motion.front();
        head_motion.pop();

        PedalMotionPoint p = pedal_motion.front();
        pedal_motion.pop();

        StateMotionPoint s = state_motion.front();
        state_motion.pop();

        std::array<double, 3> pR = b.right_position;
        std::array<double, 3> pL = b.left_position;
        double theta0 = b.waist;
        double theta7 = b.right_wrist;
        double theta8 = b.left_wrist;
        KinematicsSolver::IKResult result = solver.solve_ik(pR, pL, theta0, theta7, theta8, true);

        if (!result.success) {
            std::cerr << "[PlayMotionGenerator] PLAY: Failed to solve inverse kinematics\n";
            std::queue<std::array<double, ROBOT::NUM_JOINT>> empty_queue;
            return empty_queue;
        }

        for (int i = 0; i < 9; i++) {
            q[i] = result.q[i];   // 관절 0~8 (팔)
        }

        q[4] += s.right_elbow;
        q[6] += s.left_elbow;

        q[7] += s.right_wrist;
        q[8] += s.left_wrist;

        q[9] = p.right;
        q[10] = p.left;

        q[11] = h.yaw - q[0];
        q[12] = h.pitch;

        q_queue.push(q);
    }

    return q_queue;
}

std::pair<int, double> PlayMotionGenerator::get_num_point(double t0, double t1) {
    double n;

    // 한 라인의 데이터 개수 (5ms 단위)
    n = (t1 - t0) / ROBOT::DT_SECOND / ctx.play_speed_scale.load();
    round_sum += (int)(n * 10000) % 10000;
    if (round_sum >= 10000)
    {
        round_sum -= 10000;
        n++;
    }
    n = floor(n);

    double dt = ROBOT::DT_SECOND * ctx.play_speed_scale.load();

    return std::make_pair((int)n, dt);
}