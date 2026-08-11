#pragma once

#include <queue>
#include <mutex>
#include <vector>
#include <array>
#include <optional>

#include "common/robot_config.hpp"

enum class ControlMode {
    // T motor
    POS,
    VEL,
    
    // Maxon Motor
    CST,
    CSV,
    CSP,

    // Dynamixel or None
    NONE,
};

struct ControlSetPoint {
    std::array<double, ROBOT::NUM_JOINT> q{};
    std::array<double, ROBOT::NUM_JOINT> qd{};
    std::array<ControlMode, ROBOT::NUM_JOINT> mode{};

    bool audio_start = false;
};

class ControlQueue {
public:
    ControlQueue();
    ~ControlQueue();

    void push(const ControlSetPoint& point);
    bool empty();
    size_t size();
    std::optional<ControlSetPoint> try_pop();

private:
    std::queue<ControlSetPoint> queue_;
    std::mutex mutex_;
};