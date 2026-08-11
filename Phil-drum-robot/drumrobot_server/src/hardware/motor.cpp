#include "hardware/motor.hpp"

Motor::Motor(int id)
    : id(id) {

}

Motor::~Motor() {

}

double Motor::joint_angle_to_motor_position(double joint_angle) {
    double motor_position;
    
    motor_position = (joint_angle - initial_joint_angle) * direction_sign;

    return motor_position;
}

double Motor::motor_position_to_joint_angle(double motor_position) {
    double joint_angle;
    
    joint_angle = motor_position * direction_sign + initial_joint_angle;

    return joint_angle;
}

TMotor::TMotor(int id)
    : Motor(id) {

}

MaxonMotor::MaxonMotor(int id)
    : Motor(id) {
        
}

DynamixelMotor::DynamixelMotor(int id)
    : Motor(id) {
        
}

int32_t DynamixelMotor::angle_to_tick(double angle)
{
    double degree = angle * 180.0 / M_PI;
    degree = std::clamp(degree, -180.0, 180.0);
    const double ticks_per_degree = 4096.0 / 360.0;
    double ticks = 2048.0 - (degree * ticks_per_degree);

    return static_cast<int32_t>(std::round(ticks));
}

double DynamixelMotor::tick_to_angle(int32_t ticks)
{
    const double degrees_per_tick = 360.0 / 4096.0;
    double angle = (2048.0 - static_cast<double>(ticks)) * degrees_per_tick;
    return angle * M_PI / 180.0;
}
