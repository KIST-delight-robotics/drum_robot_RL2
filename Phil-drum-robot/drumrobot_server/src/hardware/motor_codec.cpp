#include "hardware/motor_codec.hpp"

// ===== TMotor Servo mode Codec =====
enum class CAN_PACKET_ID {
    CAN_PACKET_SET_DUTY = 0,
    CAN_PACKET_SET_CURRENT,
    CAN_PACKET_SET_DUTY_CURRENT_BRAKE,
    CAN_PACKET_SET_RPM,
    CAN_PACKET_SET_POS,
    CAN_PACKET_SET_ORIGIN_HERE,
    CAN_PACKET_SET_POS_SPD
};

std::tuple<int, float, float, float, int8_t, int8_t> TMotorServoCodec::decodeFeedback(struct can_frame *frame) {
    int id = frame->can_id & 0xFF;
    int16_t pos_int = (frame)->data[0] << 8 | (frame)->data[1];
    int16_t spd_int = (frame)->data[2] << 8 | (frame)->data[3];
    int16_t cur_int = (frame)->data[4] << 8 | (frame)->data[5];

    float pos = (float)(pos_int * 0.1f * M_PI / 180.0f); // Motor Position in radians
    float spd = (float)(spd_int * 10.0f); // Motor Speed in radians per second
    float cur = (float)(cur_int * 0.01f); // Motor Current
    int8_t temp = frame->data[6];         // Motor Temperature
    int8_t error = frame->data[7];        // Motor Error Code

    return std::make_tuple(id, pos, spd, cur, temp, error);
}

void TMotorServoCodec::encodeSetOrigin(uint32_t node_id, struct can_frame *frame, uint8_t set_origin_mode) {
    frame->can_id = node_id |
                    ((uint32_t)CAN_PACKET_ID::CAN_PACKET_SET_ORIGIN_HERE << 8 | CAN_EFF_FLAG);
    frame->can_dlc = 1;
    frame->data[0] = set_origin_mode; 
}

void TMotorServoCodec::encodePositionVelocity(uint32_t node_id, struct can_frame *frame, float pos, int16_t spd, int16_t RPA) {
    // 라디안에서 도로 변환
    float pos_deg = pos * (180.0 / M_PI);

    frame->can_id = node_id |
                    ((uint32_t)CAN_PACKET_ID::CAN_PACKET_SET_POS_SPD << 8 | CAN_EFF_FLAG);
    
    frame->can_dlc=8;
    int32_t pos_int = static_cast<int32_t>(pos_deg * 10000.0);
    frame->data[0] = (pos_int >> 24) & 0xFF;
    frame->data[1] = (pos_int >> 16) & 0xFF;
    frame->data[2] = (pos_int >> 8) & 0xFF;
    frame->data[3] = pos_int & 0xFF;

    // spd_erpm를 CAN 프레임 데이터에 저장
    frame->data[4] = (spd / 10) >> 8;
    frame->data[5] = (spd / 10) & 0xFF;

    // RPA_unit를 CAN 프레임 데이터에 저장
    frame->data[6] = (RPA / 10) >> 8;
    frame->data[7] = (RPA / 10) & 0xFF;
}

void TMotorServoCodec::encodeCurrentBrake(uint32_t node_id, struct can_frame *frame, float current) {
    frame->can_id = node_id |
                    ((uint32_t)CAN_PACKET_ID::CAN_PACKET_SET_DUTY_CURRENT_BRAKE << 8 | CAN_EFF_FLAG);
    frame->can_dlc = 4;
    int32_t current_int = static_cast<int32_t>(current * 1000.0);
    frame->data[0] = (current_int >> 24) & 0xFF;
    frame->data[1] = (current_int >> 16) & 0xFF;
    frame->data[2] = (current_int >> 8) & 0xFF;
    frame->data[3] = current_int & 0xFF;
}

void TMotorServoCodec::encodeVelocity(uint32_t node_id, struct can_frame *frame, float spd_erpm) {
    frame->can_id = node_id |
                    ((uint32_t)CAN_PACKET_ID::CAN_PACKET_SET_RPM << 8 | CAN_EFF_FLAG);
    frame->can_dlc = 4;

    // 변환된 ERPM 값을 정수로 변환
    int32_t spd_int = static_cast<int32_t>(spd_erpm);
    
    frame->data[0] = (spd_int >> 24) & 0xFF;
    frame->data[1] = (spd_int >> 16) & 0xFF;
    frame->data[2] = (spd_int >> 8) & 0xFF;
    frame->data[3] = spd_int & 0xFF;
}

void TMotorServoCodec::encodePosition(uint32_t node_id, struct can_frame *frame, float pos) {
    frame->can_id = node_id |
                    ((uint32_t)CAN_PACKET_ID::CAN_PACKET_SET_POS << 8 | CAN_EFF_FLAG);
    frame->can_dlc = 4;
    // 라디안에서 도로 변환
    float pos_deg = pos * (180.0 / M_PI);

    int32_t pos_int = static_cast<int32_t>(pos_deg*10000.0);
    
    frame->data[0] = (pos_int >> 24) & 0xFF;
    frame->data[1] = (pos_int >> 16) & 0xFF;
    frame->data[2] = (pos_int >> 8) & 0xFF;
    frame->data[3] = pos_int & 0xFF;
}

// ===== TMotor MIT mode Codec =====
void TMotorMITCodec::encodeCommand(struct can_frame *frame, int canId, int dlc, float p_des, float v_des, float kp, float kd, float t_ff) {
    // 기존 변수를 계산
    p_des = fminf(fmaxf(GLOBAL_P_MIN, p_des), GLOBAL_P_MAX);
    v_des = fminf(fmaxf(GLOBAL_V_MIN, v_des), GLOBAL_V_MAX);
    kp = fminf(fmaxf(GLOBAL_KP_MIN, kp), GLOBAL_KP_MAX);
    kd = fminf(fmaxf(GLOBAL_KD_MIN, kd), GLOBAL_KD_MAX);
    t_ff = fminf(fmaxf(GLOBAL_T_MIN, t_ff), GLOBAL_T_MAX);

    // 계산된 변수를 이용하여 unsigned int로 변환
    int p_int = floatToUint(p_des, GLOBAL_P_MIN, GLOBAL_P_MAX, 16);
    int v_int = floatToUint(v_des, GLOBAL_V_MIN, GLOBAL_V_MAX, 12);
    int kp_int = floatToUint(kp, GLOBAL_KP_MIN, GLOBAL_KP_MAX, 12);
    int kd_int = floatToUint(kd, GLOBAL_KD_MIN, GLOBAL_KD_MAX, 12);
    int t_int = floatToUint(t_ff, GLOBAL_T_MIN, GLOBAL_T_MAX, 12);
    // Set CAN frame id and data length code
    frame->can_id = canId & CAN_SFF_MASK;
    frame->can_dlc = dlc;

    /// pack ints into the can buffer ///
    frame->data[0] = p_int >> 8;                           // Position 8 higher
    frame->data[1] = p_int & 0xFF;                         // Position 8 lower
    frame->data[2] = v_int >> 4;                           // Speed 8 higher
    frame->data[3] = ((v_int & 0xF) << 4) | (kp_int >> 8); // Speed 4 bit lower KP 4bit higher
    frame->data[4] = kp_int & 0xFF;                        // KP 8 bit lower
    frame->data[5] = kd_int >> 4;                          // Kd 8 bit higher
    frame->data[6] = ((kd_int & 0xF) << 4) | (t_int >> 8); // KP 4 bit lower torque 4 bit higher
    frame->data[7] = t_int & 0xff;                         // torque 4 bit lower
}

std::tuple<int, float, float, float> TMotorMITCodec::decodeFeedback(struct can_frame *frame) {
    int id;
    float position, speed, torque;

    /// unpack ints from can buffer ///
    id = frame->data[0];
    int p_int = (frame->data[1] << 8) | frame->data[2];
    int v_int = (frame->data[3] << 4) | (frame->data[4] >> 4);
    int i_int = ((frame->data[4] & 0xF) << 8) | frame->data[5];

    /// convert ints to floats ///
    position = uintToFloat(p_int, GLOBAL_P_MIN, GLOBAL_P_MAX, 16);
    speed = uintToFloat(v_int, GLOBAL_V_MIN, GLOBAL_V_MAX, 12);
    torque = uintToFloat(i_int, -GLOBAL_I_MAX, GLOBAL_I_MAX, 12);

    return std::make_tuple(id, position, speed, torque);
}

int TMotorMITCodec::floatToUint(float x, float x_min, float x_max, unsigned int bits) {
    float span = x_max - x_min;
    if (x < x_min)
        x = x_min;
    else if (x > x_max)
        x = x_max;
    return (int)((x - x_min) * ((float)((1 << bits) / span)));
};

float TMotorMITCodec::uintToFloat(int x_int, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    return ((float)x_int) * span / ((float)((1 << bits) - 1)) + offset;
}

void TMotorMITCodec::encodeCheck(uint32_t node_id, struct can_frame *frame) {
    frame->can_id = node_id;
    frame->can_dlc = 8;

    frame->data[0] = 0x80;
    frame->data[1] = 0x00;
    frame->data[2] = 0x80;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x08;
    frame->data[7] = 0x00;
}

void TMotorMITCodec::encodeEnterControlMode(uint32_t node_id, struct can_frame *frame) {
    frame->can_id = node_id;
    frame->can_dlc = 8;

    frame->data[0] = 0xFF;
    frame->data[1] = 0xFF;
    frame->data[2] = 0xFF;
    frame->data[3] = 0xFF;
    frame->data[4] = 0xFF;
    frame->data[5] = 0xFF;
    frame->data[6] = 0xFF;
    frame->data[7] = 0xFC;
}

void TMotorMITCodec::encodeExitControlMode(uint32_t node_id, struct can_frame *frame) {
    frame->can_id = node_id;
    frame->can_dlc = 8;

    frame->data[0] = 0xFF;
    frame->data[1] = 0xFF;
    frame->data[2] = 0xFF;
    frame->data[3] = 0xFF;
    frame->data[4] = 0xFF;
    frame->data[5] = 0xFF;
    frame->data[6] = 0xFF;
    frame->data[7] = 0xFD;
}

void TMotorMITCodec::encodeSetZero(uint32_t node_id, struct can_frame *frame) {
    frame->can_id = node_id;
    frame->can_dlc = 8;

    frame->data[0] = 0xFF;
    frame->data[1] = 0xFF;
    frame->data[2] = 0xFF;
    frame->data[3] = 0xFF;
    frame->data[4] = 0xFF;
    frame->data[5] = 0xFF;
    frame->data[6] = 0xFF;
    frame->data[7] = 0xFE;
}

void TMotorMITCodec::encodeQuickStop(uint32_t node_id, struct can_frame *frame) {
    frame->can_id = node_id;
    frame->can_dlc = 8;

    frame->data[0] = 0x80;
    frame->data[1] = 0x00;
    frame->data[2] = 0x80;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x08;
    frame->data[7] = 0x00;
}

// ===== Maxon Motor Codec =====
std::tuple<int, float, float, unsigned char> MaxonMotorCodec::decodeFeedback(struct can_frame *frame) {
    int id = frame->can_id;

    unsigned char statusBit = frame->data[1];

    int32_t currentPosition = 0;
    currentPosition |= static_cast<uint8_t>(frame->data[2]);
    currentPosition |= static_cast<uint8_t>(frame->data[3]) << 8;
    currentPosition |= static_cast<uint8_t>(frame->data[4]) << 16;
    currentPosition |= static_cast<uint8_t>(frame->data[5]) << 24;

    int16_t torqueActualValue = 0;
    torqueActualValue |= static_cast<uint8_t>(frame->data[6]);
    torqueActualValue |= static_cast<uint8_t>(frame->data[7]) << 8;

    // Motor rated torque 값을 N·m 단위로 변환 (mNm -> N·m)
    const float motorRatedTorquemNm = 31.052;

    // 실제 토크 값을 N·m 단위로 계산
    // Torque actual value는 천분의 일 단위이므로, 실제 토크 값은 (torqueActualValue / 1000) * motorRatedTorqueNm
    float currentTorqueNm = (static_cast<float>(torqueActualValue) / 1000.0f) * motorRatedTorquemNm;

    float currentPositionDegrees = (static_cast<float>(currentPosition) / (35.0f * 4096.0f)) * 360.0f;
    float currentPositionRadians = currentPositionDegrees * (M_PI / 180.0f);

    return std::make_tuple(id, currentPositionRadians, currentTorqueNm, statusBit);
}

// System
void MaxonMotorCodec::encodeReadActualPos(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x40;
    frame->data[1] = 0x64;  
    frame->data[2] = 0x60;  
    frame->data[3] = 0x00;  
    frame->data[4] = 0x00;  
    frame->data[5] = 0x00;  
    frame->data[6] = 0x00;  
    frame->data[7] = 0x00;  
}

void MaxonMotorCodec::encodeCheck(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x00;
    frame->data[1] = 0x00;
    frame->data[2] = 0x00;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeStop(uint32_t node_id, struct can_frame *frame) {
    frame->can_id = 0x00;
    frame->can_dlc = 8;
    frame->data[0] = 0x02;
    frame->data[1] = node_id;
    frame->data[2] = 0x00;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeEnterOperational(uint32_t node_id, struct can_frame *frame) {
    frame->can_id = 0x00;
    frame->can_dlc = 8;
    frame->data[0] = 0x01;
    frame->data[1] = node_id;
    frame->data[2] = 0x00;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeShutdown(uint32_t tx_pdo_id_0, struct can_frame *frame) {
    frame->can_id = tx_pdo_id_0;
    frame->can_dlc = 8;
    frame->data[0] = 0x06;
    frame->data[1] = 0x00;
    frame->data[2] = 0x00;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeEnable(uint32_t tx_pdo_id_0, struct can_frame *frame) {
    frame->can_id = tx_pdo_id_0;
    frame->can_dlc = 8;
    frame->data[0] = 0x0F;
    frame->data[1] = 0x00;
    frame->data[2] = 0x00;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeSync(struct can_frame *frame) {
    frame->can_id = 0x80;
    frame->can_dlc = 0;
}

// ===== CSP mode =====
void MaxonMotorCodec::encodeCSPMode(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x60;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x08;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeTorqueOffset(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0xB2;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodePosOffset(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0xB0;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodePosition(uint32_t tx_pdo_id_1, struct can_frame *frame, float p_des_radians) {
    // 라디안 값을 인코더 값으로 변환
    float p_des_degrees = p_des_radians * (180.0f / M_PI);                        // 라디안을 도로 변환
    int p_des_enc = static_cast<int>(p_des_degrees * (35.0f * 4096.0f) / 360.0f); // 도를 인코더 값으로 변환

    unsigned char posByte0 = p_des_enc & 0xFF;
    unsigned char posByte1 = (p_des_enc >> 8) & 0xFF;
    unsigned char posByte2 = (p_des_enc >> 16) & 0xFF;
    unsigned char posByte3 = (p_des_enc >> 24) & 0xFF;

    frame->can_id = tx_pdo_id_1;
    frame->can_dlc = 4;

    frame->data[0] = posByte0;
    frame->data[1] = posByte1;
    frame->data[2] = posByte2;
    frame->data[3] = posByte3;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

// ===== HMM mode =====
void MaxonMotorCodec::encodeHomeMode(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x60;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x06;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeFollowingErrorWindow(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x65;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeHomeOffsetDistance(uint32_t can_send_id, struct can_frame *frame, int degree) {
    // 1도당 값
    float value_per_degree = 398.22;

    // 입력된 각도에 대한 값을 계산
    int value = static_cast<int>(degree * value_per_degree);

    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0xB1;
    frame->data[2] = 0x30;
    frame->data[3] = 0x00;
    frame->data[4] = value & 0xFF;
    frame->data[5] = (value >> 8) & 0xFF;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeHomePosition(uint32_t can_send_id, struct can_frame *frame, int degree) {
    float value_per_degree = 398.22;
    int32_t value = static_cast<int32_t>(degree * value_per_degree);

    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0xB0;
    frame->data[2] = 0x30;
    frame->data[3] = 0x00;
    frame->data[4] = value & 0xFF;
    frame->data[5] = (value >> 8) & 0xFF;
    frame->data[6] = (value >> 16) & 0xFF;
    frame->data[7] = (value >> 24) & 0xFF;
}

void MaxonMotorCodec::encodeHomingMethodLeft(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x98;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x25;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;

    // encodeHomingMethodRight 동일한 함수
    // 현재 사용하지 않는 기능
}

void MaxonMotorCodec::encodeHomingMethodRight(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x98;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x25;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;

    // encodeHomingMethodLeft 이랑 동일한 함수
    // 현재 사용하지 않는 기능
}

void MaxonMotorCodec::encodeHomingMethodTest(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x98;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x25;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeStartHoming(uint32_t tx_pdo_id_0, struct can_frame *frame) {
    frame->can_id = tx_pdo_id_0;
    frame->can_dlc = 8;
    frame->data[0] = 0x1F;
    frame->data[1] = 0x00;
    frame->data[2] = 0x00;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeCurrentThresholdRight(uint32_t can_send_id, struct can_frame *frame) {
    // 1000 = 3E8
    // 500 = 01F4
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x23;
    frame->data[1] = 0xB2;
    frame->data[2] = 0x30;
    frame->data[3] = 0x00;
    frame->data[4] = 0xF4;
    frame->data[5] = 0x01;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeCurrentThresholdLeft(uint32_t can_send_id, struct can_frame *frame) {
    // 1000 =03E8
    // 500 = 01F4
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x23;
    frame->data[1] = 0xB2;
    frame->data[2] = 0x30;
    frame->data[3] = 0x00;
    frame->data[4] = 0xF4;
    frame->data[5] = 0x01;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

// ===== CSV mode =====
void MaxonMotorCodec::encodeCSVMode(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x60;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x09;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeVelOffset(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0xB1;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeTargetVelocity(uint32_t tx_pdo_id_2, struct can_frame *frame, int targetVelocity) {
    unsigned char velByte0 = targetVelocity & 0xFF;
    unsigned char velByte1 = (targetVelocity >> 8) & 0xFF;
    unsigned char velByte2 = (targetVelocity >> 16) & 0xFF;
    unsigned char velByte3 = (targetVelocity >> 24) & 0xFF;

    frame->can_id = tx_pdo_id_2;
    frame->can_dlc = 4;

    frame->data[0] = velByte0;
    frame->data[1] = velByte1;
    frame->data[2] = velByte2;
    frame->data[3] = velByte3;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

// ===== CST mode =====
void MaxonMotorCodec::encodeCSTMode(uint32_t can_send_id, struct can_frame *frame) {
    frame->can_id = can_send_id;
    frame->can_dlc = 8;
    frame->data[0] = 0x22;
    frame->data[1] = 0x60;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = 0x0A;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeTorque(uint32_t tx_pdo_id_3, struct can_frame *frame, int targetTorquemNm) {
    // Motor rated torque 값 (mN·m)
    const float motorRatedTorquemNm = 31.052;
    const float maxTorquemNm = 293.8;

    int targetTorque;

    // saturation
    if (targetTorquemNm > maxTorquemNm) {
        targetTorque = static_cast<int>(maxTorquemNm/motorRatedTorquemNm*1000.0);
    } else if (targetTorquemNm < -1.0*maxTorquemNm) {
        targetTorque = static_cast<int>(-1.0*maxTorquemNm/motorRatedTorquemNm*1000.0);
    } else {
        targetTorque = static_cast<int>(targetTorquemNm/motorRatedTorquemNm*1000.0);
    }

    unsigned char torByte0 = targetTorque & 0xFF;
    unsigned char torByte1 = (targetTorque >> 8) & 0xFF;

    frame->can_id = tx_pdo_id_3;
    frame->can_dlc = 2;

    frame->data[0] = torByte0;
    frame->data[1] = torByte1;
    frame->data[2] = 0x00;
    frame->data[3] = 0x00;
    frame->data[4] = 0x00;
    frame->data[5] = 0x00;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}

void MaxonMotorCodec::encodeTorqueSDO(uint32_t can_send_id, struct can_frame *frame, int targetTorque) {
    unsigned char torByte0 = targetTorque & 0xFF;
    unsigned char torByte1 = (targetTorque >> 8) & 0xFF;

    frame->can_id = can_send_id;
    frame->can_dlc = 8;

    frame->data[0] = 0x2B;
    frame->data[1] = 0x71;
    frame->data[2] = 0x60;
    frame->data[3] = 0x00;
    frame->data[4] = torByte0;
    frame->data[5] = torByte1;
    frame->data[6] = 0x00;
    frame->data[7] = 0x00;
}