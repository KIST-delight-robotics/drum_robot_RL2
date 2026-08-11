#pragma once

#include <array>
#include <string>
#include <unordered_map>

namespace ROBOT {
    inline constexpr int    NUM_JOINT      = 13;
    inline constexpr int    NUM_INSTRUMENT = 10;
    inline constexpr double DT_SECOND      = 0.005;
}

struct InstrumentCoordinate {
    // 드럼 위치
    // 드럼을 치는 순간 손목 각도
    std::array<double, 3> right_position;
    double                right_wrist_angle;
    std::array<double, 3> left_position;
    double                left_wrist_angle;
};

static const std::unordered_map<std::string, int> instrument_name_to_id = {
    {"bass",         0},
    {"snare",        1},
    {"floor",        2},
    {"mid",          3},
    {"top",          4},
    {"closed hihat", 5},
    {"ride",         6},
    {"right crash",  7},
    {"left crash",   8},
    {"open hihat",   9},
};