#pragma once

#include <queue>
#include <vector>
#include <array>
#include <map>

#include "common/app_context.hpp"
#include "common/motion_queue.hpp"
#include "common/robot_config.hpp"
#include "kinematics/kinematics_solver.hpp"
#include "trajectory/base_motion_generator.hpp"
#include "trajectory/head_motion_generator.hpp"
#include "trajectory/pedal_motion_generator.hpp"
#include "trajectory/state_motion_generator.hpp"
#include "util/logger.hpp"

class PlayMotionGenerator {
public:
    PlayMotionGenerator(AppContext &ctxRef);
    ~PlayMotionGenerator();

    void initialize();
    bool reset(std::array<double, ROBOT::NUM_JOINT>& q, int note_r = 1, int note_l = 1);    // 초기 위치 기본값: 스네어

    std::queue<std::array<double, ROBOT::NUM_JOINT>> generate_motion(const std::vector<DrumEvent>& rds);

private:
    AppContext &ctx;
    
    KinematicsSolver solver;

    std::map<int, InstrumentCoordinate> drum_coordinates;

    BaseMotionGenerator base_motion_generator;
    HeadMotionGenerator head_motion_generator;
    PedalMotionGenerator pedal_motion_generator;
    StateMotionGenerator state_motion_generator;

    std::pair<int, double> get_num_point(double t0, double t1);

    int round_sum = 0;      // 소수점 오차 보정
};