#pragma once

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include <queue>
#include <array>

#include "common/app_context.hpp"
#include "common/control_queue.hpp"
#include "common/motion_queue.hpp"  // MotionPrimitive
#include "common/robot_config.hpp"
#include "kinematics/kinematics_solver.hpp"
#include "trajectory/play_motion_generator.hpp"
#include "util/logger.hpp"

class TrajectoryGenerator {
public:
    TrajectoryGenerator(AppContext& ctxRef, ControlQueue &controlQueueRef);
    ~TrajectoryGenerator();

    void initialize(const std::map<std::string, std::vector<double>>& pose);
    void generate_trajectory(const MotionPrimitive& motion);
 
private:
    AppContext &ctx;
    ControlQueue &control_queue;

    KinematicsSolver solver;
    PlayMotionGenerator play_motion_generator;

    std::array<double, ROBOT::NUM_JOINT> last_q;     // 마지막 위치
    std::array<double, ROBOT::NUM_JOINT> last_qd;    // 마지막 속도

    std::array<double, 3> last_p_R;   // 마지막 위치
    std::array<double, 3> last_p_L;   // 마지막 위치

    ControlMode tmotor_control_mode = ControlMode::VEL;
    ControlMode wrist_control_mode = ControlMode::CSP;
    ControlMode pedal_control_mode = ControlMode::CSP;
    
    void generate_standby_trajectory();
    void generate_joint_space_trajectory(const MotionPrimitive& motion);
    void generate_task_space_trajectory(const MotionPrimitive& motion);
    void generate_play_start_trajectory(const MotionPrimitive& motion);
    void generate_play_end_trajectory();
    void generate_play_trajectory(const MotionPrimitive& motion);
    void generate_idle_trajectory();
    void generate_replay_trajectory(const MotionPrimitive& motion);   // Stage A: CSV 오픈루프 재생

    std::vector<double> ready_pose;   // play 후 돌아오는 대기 자세 (다음 동작 준비 상태)

    std::array<ControlMode, ROBOT::NUM_JOINT> get_modes(bool is_play = false);
    std::pair<std::vector<double>, std::vector<double>> sample(
        const std::vector<double>& q0,
        const std::vector<double>& q1,
        int n,
        int k,
        TrajectoryProfile profile
    );
    std::pair<std::vector<double>, std::vector<double>> sample_trapezoidal(
        const std::vector<double>& q0,
        const std::vector<double>& q1,
        int n,
        int k
    );
    std::pair<std::vector<double>, std::vector<double>> sample_cubic(
        const std::vector<double>& q0,
        const std::vector<double>& q1,
        int n,
        int k
    );
    std::pair<std::vector<double>, std::vector<double>> sample_quintic(
        const std::vector<double>& q0,
        const std::vector<double>& q1,
        int n,
        int k
    );
    std::pair<std::vector<double>, std::vector<double>> sample_cosine(
        const std::vector<double>& q0,
        const std::vector<double>& q1,
        int n,
        int k
    );
    void update_last_q(const std::vector<double>& q);
    void update_last_q(const std::array<double, ROBOT::NUM_JOINT>& q);
    void update_last_q(const std::vector<double>& p, const std::vector<double>& q);

    // ===== log =====
    Logger trajectory_log;

    // ===== audio =====
    bool first_point = false;
};