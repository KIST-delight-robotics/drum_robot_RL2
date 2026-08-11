#pragma once

#include <queue>
#include <vector>
#include <cmath>

#include "common/motion_queue.hpp"
#include "common/robot_config.hpp"

struct StateMotionPoint {
    double right_elbow;
    double left_elbow;

    double right_wrist;
    double left_wrist;
};

class StateMotionGenerator {
public:
    StateMotionGenerator();
    ~StateMotionGenerator();

    StateMotionPoint reset();
    std::queue<StateMotionPoint> generate_motion(const std::vector<DrumEvent> rds, int num_point, double dt);
    bool get_error();

private:
    enum class Arm { RIGHT, LEFT };

    enum class State {
        REST_TO_REST,    // 타격 없음 -> 타격 없음
        HIT_TO_REST,     // 타격 후 -> 다음 없음
        REST_TO_HIT,     // 대기 후 -> 다음 타격
        HIT_TO_HIT       // 타격 후 -> 다음 타격
    };

    const double HIT_DETECTION_THRESHOLD = 0.5;     // 다음 타격 감지할 최대 시간 [s]
    const double BEAT_STEP = 0.05;                  // 악보를 쪼개는 단위

    struct MotionContext {
        double last_t = 0.0;            // 이전 타격 시간
        State  state  = State::REST_TO_REST;
        int    intensity = 0;           // 이전 타격 강도
    };

    // 한 팔의 타격 구간 정보
    struct HitSegment {
        double start_time;
        double end_time;
        State  state;
        int    intensity;
        MotionContext next_context;
    };

    struct SubLine {
        double t;
        int vel_R = 0;
        int vel_L = 0;
    };

    std::vector<SubLine> split_line(const std::vector<DrumEvent>& rds);

    // 시간 파라미터
    struct ElbowTime {
        double t_lift;
        double t_hit;
    };
    struct WristTime {
        double t_stay;
        double t_lift;
        double t_hit;
    };

    // 각도 파라미터
    struct ElbowAngle {
        double stay = 0.0;
        double lift = 15.0 * M_PI / 180.0;
    };
    struct WristAngle {
        double stay  =  20.0 * M_PI / 180.0;
        double press =  -5.0 * M_PI / 180.0;
        double lift  =  40.0 * M_PI / 180.0;
    };

    MotionContext right_context;
    MotionContext left_context;

    HitSegment get_hit_segment(const std::vector<SubLine>& sub, int idx, Arm arm);

    ElbowTime  get_elbow_time(double t1, double t2, int intensity);
    WristTime  get_wrist_time(double t1, double t2, int intensity, State state);
    ElbowAngle get_elbow_angle_param(double t1, double t2, int intensity);
    WristAngle get_wrist_angle_param(double t1, double t2, int intensity);

    double get_elbow_angle(double t, const HitSegment& seg);
    double get_wrist_angle(double t, const HitSegment& seg);

    // 닫힌 형태 보간
    static double cubic_hermite(double ta, double qa, double va, double tb, double qb, double vb, double t);
    // zero_vel_at_start == true  : q'(ta)=0 으로 결정되는 2차 (출발 속도 0)
    // zero_vel_at_start == false : q'(tb)=0 으로 결정되는 2차 (도착 속도 0)
    static double quadratic(double ta, double qa, double tb, double qb, bool zero_vel_at_start, double t);

    bool state_end_error = false;
};