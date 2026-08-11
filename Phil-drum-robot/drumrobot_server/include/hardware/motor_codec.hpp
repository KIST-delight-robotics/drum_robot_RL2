#pragma once

#include <linux/can.h>
#include <cmath>
#include <tuple>
#include <iostream>

using namespace std;

class TMotorServoCodec {
public:
    // node_id: motor.node_id, 0: tempoaray origin  1: setting a permanent zero point
    void encodeSetOrigin(uint32_t node_id, struct can_frame *frame, uint8_t set_origin_mode);
    
    // node_id: motor.node_id, pos [rad], speed[erpm], acceleration [erpm/s^2]
    void encodePositionVelocity(uint32_t node_id, struct can_frame *frame, float pos, int16_t spd, int16_t RPA);

    // id, position, speed, current, temperature, error
    std::tuple<int, float, float, float, int8_t, int8_t> decodeFeedback(struct can_frame *frame);

    // node_id: motor.node_id, current break
    void encodeCurrentBrake(uint32_t node_id, struct can_frame *frame, float current);

    // node_id: motor.node_id, speed [erpm]
    void encodeVelocity(uint32_t node_id, struct can_frame *frame, float spd_erpm);

    // node_id: motor.node_id, pos [rad]
    void encodePosition(uint32_t node_id, struct can_frame *frame, float pos);

private:
};

class TMotorMITCodec {
public:
    float GLOBAL_P_MIN = -12.5;
    float GLOBAL_P_MAX = 12.5;
    float GLOBAL_KP_MIN = 0;
    float GLOBAL_KP_MAX = 500;
    float GLOBAL_KD_MIN = 0;
    float GLOBAL_KD_MAX = 5;
    float GLOBAL_V_MIN, GLOBAL_V_MAX, GLOBAL_T_MIN, GLOBAL_T_MAX;
    float GLOBAL_I_MAX, Kt;


    void encodeCommand(struct can_frame *frame, int canId, int dlc, float p_des, float v_des, float kp, float kd, float t_ff);
    std::tuple<int, float, float, float> decodeFeedback(struct can_frame *frame);

    // node_id: motor.node_id
    void encodeCheck(uint32_t node_id, struct can_frame *frame);

    // node_id: motor.node_id
    void encodeEnterControlMode(uint32_t node_id, struct can_frame *frame);

    // node_id: motor.node_id
    void encodeExitControlMode(uint32_t node_id, struct can_frame *frame);

    // node_id: motor.node_id
    void encodeSetZero(uint32_t node_id, struct can_frame *frame);

    // node_id: motor.node_id
    void encodeQuickStop(uint32_t node_id, struct can_frame *frame);

private:
    int floatToUint(float x, float x_min, float x_max, unsigned int bits);
    float uintToFloat(int x_int, float x_min, float x_max, int bits);
};

class MaxonMotorCodec {
public:
    std::tuple<int, float, float, unsigned char> decodeFeedback(struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeReadActualPos(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeCheck(uint32_t can_send_id, struct can_frame *frame);

    // node_id: motor.node_id
    void encodeStop(uint32_t node_id, struct can_frame *frame);

    // node_id: motor.node_id
    void encodeEnterOperational(uint32_t node_id, struct can_frame *frame);

    // tx_pdo_id_0: motor.tx_pdo_ids[0]
    void encodeShutdown(uint32_t tx_pdo_id_0, struct can_frame *frame);

    // tx_pdo_id_0: motor.tx_pdo_ids[0]
    void encodeEnable(uint32_t tx_pdo_id_0, struct can_frame *frame);

    void encodeSync(struct can_frame *frame);

    // ====== CSP 모드 관련 명령 구성 메서드 ======

    // can_send_id: motor.can_send_id
    void encodeCSPMode(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeTorqueOffset(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodePosOffset(uint32_t can_send_id, struct can_frame *frame);

    // tx_pdo_id_1: motor.tx_pdo_ids[1]
    void encodePosition(uint32_t tx_pdo_id_1, struct can_frame *frame, float p_des_radians);

    // ====== HMM 모드 관련 명령 구성 메서드 ======
    
    // can_send_id: motor.can_send_id
    void encodeHomeMode(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeFollowingErrorWindow(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeHomeOffsetDistance(uint32_t can_send_id, struct can_frame *frame, int degree);

    // motor.can_send_id
    void encodeHomePosition(uint32_t can_send_id, struct can_frame *frame, int degree);

    // can_send_id: motor.can_send_id
    void encodeHomingMethodLeft(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeHomingMethodRight(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeHomingMethodTest(uint32_t can_send_id, struct can_frame *frame);

    // tx_pdo_id_0: motor.tx_pdo_ids[0]
    void encodeStartHoming(uint32_t tx_pdo_id_0, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeCurrentThresholdRight(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeCurrentThresholdLeft(uint32_t can_send_id, struct can_frame *frame);

    // ===== CSV 모드 관련 명령 구성 메서드 ======

    // can_send_id: motor.can_send_id
    void encodeCSVMode(uint32_t can_send_id, struct can_frame *frame);

    // can_send_id: motor.can_send_id
    void encodeVelOffset(uint32_t can_send_id, struct can_frame *frame);

    // tx_pdo_id_2: motor.tx_pdo_ids[2]
    void encodeTargetVelocity(uint32_t tx_pdo_id_2, struct can_frame *frame, int targetVelocity);

    // ====== CST 모드 관련 명령 구성 메서드 ======

    // can_send_id: motor.can_send_id
    void encodeCSTMode(uint32_t can_send_id, struct can_frame *frame);

    // tx_pdo_id_3: motor.tx_pdo_ids[3]
    void encodeTorque(uint32_t tx_pdo_id_3, struct can_frame *frame, int targetTorquemNm);

    // can_send_id: motor.can_send_id
    void encodeTorqueSDO(uint32_t can_send_id, struct can_frame *frame, int targetTorque);
};