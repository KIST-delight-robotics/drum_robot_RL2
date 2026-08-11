#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstring>
#include <utility>

#include "dynamixel_sdk.h"

#include "common/control_queue.hpp"

class Motor{
public:
    Motor(int id);
    virtual ~Motor();

    int id;
    std::string name;

    double direction_sign;
    double initial_joint_angle;
    double min_angle;
    double max_angle;
    double motor_min;
    double motor_max;

    double current_joint_angle = 0.0;

    double joint_angle_to_motor_position(double joint_angle);
    double motor_position_to_joint_angle(double motor_position);
private:
};

class TMotor : public Motor {
public:
    TMotor(int id);

    // For CAN communication
    uint32_t node_id;
    int socket;
    bool is_connected = false;

    std::string model;          // 모델
    const double pole = 21.0;   // 극수
    double gear_ratio;          // 기어비
    double current_limit;       // 전류 제한
    double control_gain;

    double current_position = 0.0;
    double current_velocity = 0.0;
    double current_motor_current  = 0.0;

    // 과전류 체크 카운터
    int cnt = 0;

    // 수신이 한 번이라도 되었는지 확인
    bool first_recv_done = false;
private:
};

class MaxonMotor : public Motor {
public:
    MaxonMotor(int id);

    // For CAN communication
    uint32_t node_id;
    int socket;
    bool is_connected = false;

    uint32_t can_send_id;
    uint32_t can_receive_id;

    uint32_t tx_pdo_ids[4];
    uint32_t rx_pdo_ids[2];

    double gear_ratio;
    double control_kp;          // CST 모드용 P 게인
    double control_kd;          // CST 모드용 D 게인
    double prev_err = 0.0;       // CST 모드용 이전 오차
    double prev_err_dot = 0.0;   // CST 모드용 이전 오차 미분

    ControlMode mode = ControlMode::CSP;

    double current_position = 0.0;
    double current_torque   = 0.0;
    unsigned char status_bit = 0;
private:
};

class DynamixelMotor : public Motor {
public:
    DynamixelMotor(int id);

    uint8_t dxl_id;

    int32_t angle_to_tick(double angle);
    double tick_to_angle(int32_t ticks);

    double current_position = 0.0;

private:
};