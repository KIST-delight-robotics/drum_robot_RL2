#pragma once
 
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <set>
#include <string>
 
#include "common/app_context.hpp"
#include "common/control_queue.hpp"
#include "common/robot_config.hpp"
#include "hardware/motor_codec.hpp"
#include "hardware/robot.hpp"
#include "util/audio_player.hpp"
#include "util/logger.hpp"
 
class Controller {
public:
    Controller(AppContext &ctxRef, ControlQueue &controlQueueRef, Robot &robotRef, AudioPlayer &audioRef);
    ~Controller();
 
    void send_loop();
    void recv_loop();
 
private:
    AppContext &ctx;
    ControlQueue &control_queue;
    Robot &robot;
 
    TMotorServoCodec t_codec;
    MaxonMotorCodec m_codec;
 
    ControlSetPoint curr_point; // 현재 송신 중인 데이터 (5ms 주기)
    ControlSetPoint prev_point; // 맥슨 모터 보간 목표
 
    // ===== SEND =====
    void send_task_1ms(int cnt);
    void send_task_5ms();

    bool all_tmotors_received();

    void tmotor_send_task(const ControlSetPoint &point);
    void maxon_motor_send_task(const ControlSetPoint &point);
    void dynamixel_send_task(const ControlSetPoint &point);

    void set_maxon_mode(std::shared_ptr<MaxonMotor> &maxon, ControlMode target_mode);
    double cal_torque(std::shared_ptr<MaxonMotor> &maxon, double target_position);

    const double STICK_LEN_M   = 0.121;
    const double STICK_MASS_KG = 0.0845;

    const double POS_DIFF_LIMIT = 30.0f * M_PI / 180.0f; // 30도
 
    // ===== RECV =====
    void read_frames(); // 소켓에서 프레임을 논블록으로 읽어 tempFrames 에 누적
    void distribute_frames();   // can frame을 파싱해 각 모터 상태 업데이트
 
    bool safety_check_recv_tmotor(std::shared_ptr<TMotor> &motor);
    bool safety_check_recv_maxon(std::shared_ptr<MaxonMotor> &motor);
 
    // 소켓별 수신 프레임 임시 버퍼  (socket_fd, frames)
    std::map<int, std::vector<struct can_frame>> temp_frames;

    // ===== log =====
    Logger motor_log;

    // ===== audio =====
    AudioPlayer &audio_player;
};