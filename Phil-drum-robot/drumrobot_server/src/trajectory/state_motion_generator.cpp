#include "trajectory/state_motion_generator.hpp"

#include <algorithm>
#include <iostream>

StateMotionGenerator::StateMotionGenerator() {

}

StateMotionGenerator::~StateMotionGenerator() {

}

StateMotionPoint StateMotionGenerator::reset() {
    right_context = MotionContext{};
    left_context  = MotionContext{};

    StateMotionPoint point;

    ElbowAngle e;
    point.right_elbow = e.stay;
    point.left_elbow = e.stay;

    WristAngle w;
    point.right_wrist = w.stay;
    point.left_wrist = w.stay;

    state_end_error = false;

    return point;
}

std::queue<StateMotionPoint> StateMotionGenerator::generate_motion(const std::vector<DrumEvent> rds, int num_point, double dt) {
    std::queue<StateMotionPoint> out;

    if (rds.size() < 2 || num_point <= 0) {
        return out;
    }

    // rds를 0.05초 단위로 쪼갬
    std::vector<SubLine> sub = split_line(rds);
    if (sub.empty()) {
        return out;
    }

    int sub_line = 0;
    double t0 = rds[0].t;
    int samples = static_cast<int>(round(rds[1].beat / BEAT_STEP));
    HitSegment seg_R, seg_L;

    // std::cout << "[StateMotion] samples=" << samples << "\n";
    // for (int i = 0; i < (int)sub.size(); i++) {
    //     std::cout << "[" << i << "] t: " << sub[i].t
    //               << "  vel_R: " << sub[i].vel_R
    //               << "  vel_L: " << sub[i].vel_L << "\n";
    // }

    for (int i = 0; i < num_point; i++) {
        if (i >= sub_line*num_point/samples) {
            seg_R = get_hit_segment(sub, sub_line, Arm::RIGHT);
            seg_L = get_hit_segment(sub, sub_line, Arm::LEFT);

            right_context = seg_R.next_context;
            left_context = seg_L.next_context;

            // auto state_str = [](State s) -> const char* {
            //     switch (s) {
            //         case State::REST_TO_REST: return "REST_TO_REST";
            //         case State::HIT_TO_REST:  return "HIT_TO_REST";
            //         case State::REST_TO_HIT:  return "REST_TO_HIT";
            //         case State::HIT_TO_HIT:   return "HIT_TO_HIT";
            //         default:                  return "UNKNOWN";
            //     }
            // };

            // std::cout << "[StateMotion] sub_line=" << sub_line
            //           << "  i=" << i
            //           << "\n  R: state=" << state_str(seg_R.state)
            //           << " intensity=" << seg_R.intensity
            //           << " start=" << seg_R.start_time
            //           << " end=" << seg_R.end_time
            //           << "\n  L: state=" << state_str(seg_L.state)
            //           << " intensity=" << seg_L.intensity
            //           << " start=" << seg_L.start_time
            //           << " end=" << seg_L.end_time
            //           << "\n";

            sub_line++;
        }

        StateMotionPoint point;

        double t_now = i * dt + t0;
        double t_R = t_now - seg_R.start_time;
        double t_L = t_now - seg_L.start_time;

        point.right_elbow = get_elbow_angle(t_R, seg_R);
        point.left_elbow  = get_elbow_angle(t_L, seg_L);

        point.right_wrist = get_wrist_angle(t_R, seg_R);
        point.left_wrist  = get_wrist_angle(t_L, seg_L);

        out.push(point);
    }

    return out;
}

bool StateMotionGenerator::get_error() {
    return state_end_error;
}

std::vector<StateMotionGenerator::SubLine> StateMotionGenerator::split_line(const std::vector<DrumEvent>& rds) {
    // 생성 대상은 첫 줄 (rds[0] -> rds[1]) 하나.
    // 단, 마지막 조각에서 다음 타격을 (0.5초 윈도우로) 탐색하려면
    // rds[1] 이후 줄들도 함께 쪼개 둬야 함 (과거 divideMatrix 가 measureMatrix 전체를 쪼갠 것과 동일).
    // 포인트 생성은 첫 줄 조각만 사용하고, 그 뒤 조각은 탐색용.
    std::vector<SubLine> sub;
    int rds_size = (int)rds.size();

    // 첫 조각 = 시작 상태 (rds[0])
    SubLine first;
    first.t = rds[0].t;
    first.vel_R = rds[0].velocity_R;
    first.vel_L = rds[0].velocity_L;
    sub.push_back(first);

    // rds[1..] 각 줄을 0.05초 단위로 쪼갬
    for (int line = 1; line < rds_size; line++) {
        int steps = (int)round(rds[line].beat / BEAT_STEP);
        if (steps < 1) {
            steps = 1;
        }
        double line_time = rds[line].t - rds[line - 1].t;
        double step_dt = line_time / steps;

        for (int s = 0; s < steps; s++) {
            SubLine sl;
            sl.t   = rds[line - 1].t + (s + 1) * step_dt;

            if (s + 1 == steps) {   // 줄의 마지막 조각에만 타격 정보
                sl.vel_R = rds[line].velocity_R;
                sl.vel_L = rds[line].velocity_L;
            }

            sub.push_back(sl);
        }

        // 첫 줄 + 0.5초 윈도우면 탐색에 충분 -> 더 이상 쪼개지 않음
        if (round(10000 * (rds[line].t - rds[1].t)) > round(10000 * HIT_DETECTION_THRESHOLD)) {
            break;
        }
    }

    return sub;
}

StateMotionGenerator::HitSegment StateMotionGenerator::get_hit_segment(const std::vector<SubLine>& sub, int idx, Arm arm) {
    HitSegment seg;
    int n = (int)sub.size();
    if (idx + 1 >= n) {
        state_end_error = true;
        return seg;   // 인덱싱 오류
    }

    const MotionContext& ctx = (arm == Arm::RIGHT) ? right_context : left_context;
    auto vel_of  = [&](int i) { return (arm == Arm::RIGHT) ? sub[i].vel_R  : sub[i].vel_L;  };

    double t_cur = sub[idx].t;

    // ===== 다음 타격 감지 (현재 조각 이후) =====
    bool is_hit = false;
    double t_hit = sub[idx + 1].t;
    int hit_vel = 0;

    for (int i = idx + 1; i < n; i++) {
        if (round(10000 * HIT_DETECTION_THRESHOLD) < round(10000 * (sub[i].t - t_cur))) {
            if (i != idx + 1) {     // 첫 줄은 무조건 읽도록
                break;
            }
        }

        if (vel_of(i) != 0) {
            is_hit  = true;
            t_hit   = sub[i].t;
            hit_vel = vel_of(i);
            break;
        }
    }

    // ===== state, 시간, intensity 결정 =====
    MotionContext next;

    if (vel_of(idx) == 0) {
        // 현재 조각이 타격이 아님
        if (ctx.state == State::REST_TO_HIT || ctx.state == State::HIT_TO_HIT) {
            // 이전에 타격 O -> 이어서 이동
            seg.start_time = ctx.last_t;
            seg.end_time   = t_hit;
            seg.state      = ctx.state;
            seg.intensity  = ctx.intensity;

            next = ctx;
        } else {
            if (is_hit) {
                seg.start_time = t_cur;
                seg.end_time   = t_hit;
                seg.state      = State::REST_TO_HIT;
                seg.intensity  = hit_vel;

                next.last_t    = t_cur;
                next.state     = State::REST_TO_HIT;
                next.intensity = hit_vel;
            } else {
                seg.start_time = t_cur;
                seg.end_time   = sub[idx + 1].t;
                seg.state      = State::REST_TO_REST;
                seg.intensity  = 0;

                next.last_t    = t_cur;
                next.state     = State::REST_TO_REST;
                next.intensity = 0;
            }
        }
    } else {
        // 현재 조각이 타격임
        if (is_hit) {
            seg.start_time = t_cur;
            seg.end_time   = t_hit;
            seg.state      = State::HIT_TO_HIT;
            seg.intensity  = hit_vel;

            next.last_t    = t_cur;
            next.state     = State::HIT_TO_HIT;
            next.intensity = hit_vel;
        } else {
            seg.start_time = t_cur;
            seg.end_time   = sub[idx + 1].t;
            seg.state      = State::HIT_TO_REST;
            seg.intensity  = vel_of(idx);

            next.last_t    = t_cur;
            next.state     = State::HIT_TO_REST;
            next.intensity = vel_of(idx);
        }
    }

    seg.next_context = next;

    return seg;
}

StateMotionGenerator::ElbowTime StateMotionGenerator::get_elbow_time(double t1, double t2, int intensity) {
    (void)intensity;
    ElbowTime et;
    double T = t2 - t1;     // 전체 타격 시간

    et.t_lift = std::max(0.5 * T, T - 0.2);
    et.t_hit  = T;

    return et;
}

StateMotionGenerator::WristTime StateMotionGenerator::get_wrist_time(double t1, double t2, int intensity, State state) {
    (void)intensity;
    WristTime wt;
    double T = t2 - t1;     // 전체 타격 시간

    if (state == State::REST_TO_HIT) {
        wt.t_lift = std::max(0.7 * T, T - 0.2);     // 최고점에 도달하는 시간
        wt.t_stay = 0.4 * T;                        // 상승하기 시작하는 시간
    } else {
        // HIT_TO_REST or HIT_TO_HIT (복귀 모션 필요)
        if (T <= 0.5) {
            wt.t_lift = 0.5 * T;
        } else {
            wt.t_lift = T - 0.25;
        }
        wt.t_stay = 0.5 * T;
    }

    wt.t_hit = T;

    return wt;
}

StateMotionGenerator::ElbowAngle StateMotionGenerator::get_elbow_angle_param(double t1, double t2, int intensity) {
    ElbowAngle ea;
    double T = t2 - t1;     // 전체 타격 시간

    double intensity_factor;    // 1~3: 0%, 4: 90%, 5: 100%, 6: 110%, 7: 120%
    if (intensity <= 3) {
        intensity_factor = 0.0;
    } else {
        intensity_factor = 0.1 * intensity + 0.5;
    }

    double lift;
    if (T < 0.2) {
        lift = 0.0;                                 // 0.2초보다 짧을 땐 안 들게
    } else if (T <= 0.5) {
        lift = T * (15.0 * M_PI / 180.0) / 0.5;     // 0.2~0.5초: 선형 감소
    } else {
        lift = 15.0 * M_PI / 180.0;
    }

    ea.lift = ea.stay + lift * intensity_factor;

    return ea;
}

StateMotionGenerator::WristAngle StateMotionGenerator::get_wrist_angle_param(double t1, double t2, int intensity) {
    WristAngle wa;
    double T = t2 - t1;     // 전체 타격 시간

    double intensity_factor;
    if (intensity < 5) {
        intensity_factor = 0.25 * intensity - 0.25;        // 1:0% 2:25% 3:50% 4:75%
    } else {
        intensity_factor = intensity / 3.0 - (2.0 / 3.0);  // 5:100% 6:133% 7:167%
    }

    // Lift Angle (최고점 각도), 최대 40도 * 세기
    double lift = (T < 0.5) ? (40.0 * T + 10.0) * M_PI / 180.0 : 30.0 * M_PI / 180.0;

    if (intensity == 1) {
        wa.lift = wa.stay;          // intensity 1일 땐 아예 안 들도록
    } else {
        wa.lift = wa.stay + lift * intensity_factor;
    }

    return wa;
}

double StateMotionGenerator::get_elbow_angle(double t, const HitSegment& seg) {
    ElbowTime  et = get_elbow_time(seg.start_time, seg.end_time, seg.intensity);
    ElbowAngle ea = get_elbow_angle_param(seg.start_time, seg.end_time, seg.intensity);

    switch (seg.state) {
        case State::REST_TO_REST:
            // Stay
            return ea.stay;

        case State::HIT_TO_REST: {
            // Release : 0 -> stay (양끝 속도 0)
            if (ea.stay == 0.0) {
                return 0.0;
            }
            return cubic_hermite(0.0, 0.0, 0.0, et.t_hit, ea.stay, 0.0, t);
        }

        case State::REST_TO_HIT: {
            if (ea.lift == ea.stay) {
                // 드는 궤적 없음
                return (t < et.t_lift) ? ea.stay : 0.0;
            }
            if (t < et.t_lift) {
                // Lift : stay -> lift
                return cubic_hermite(0.0, ea.stay, 0.0, et.t_lift, ea.lift, 0.0, t);
            } else {
                // Hit : lift -> 0
                return cubic_hermite(et.t_lift, ea.lift, 0.0, et.t_hit, 0.0, 0.0, t);
            }
        }

        case State::HIT_TO_HIT: {
            if (ea.lift == ea.stay) {
                return 0.0;
            }
            if (t < et.t_lift) {
                // Lift : 0 -> lift
                return cubic_hermite(0.0, 0.0, 0.0, et.t_lift, ea.lift, 0.0, t);
            } else {
                // Hit : lift -> 0
                return cubic_hermite(et.t_lift, ea.lift, 0.0, et.t_hit, 0.0, 0.0, t);
            }
        }
    }

    return ea.stay;
}

double StateMotionGenerator::get_wrist_angle(double t, const HitSegment& seg) {
    WristTime  wt = get_wrist_time(seg.start_time, seg.end_time, seg.intensity, seg.state);
    WristAngle wa = get_wrist_angle_param(seg.start_time, seg.end_time, seg.intensity);

    switch (seg.state) {
        case State::REST_TO_REST:
            // Stay
            return wa.stay;

        case State::HIT_TO_REST: {
            // Release : press -> stay (도착 속도 0), 2차
            if (wa.press == wa.stay) {
                return wa.stay;
            }
            return quadratic(0.0, wa.press, wt.t_hit, wa.stay, false, t);
        }

        case State::REST_TO_HIT: {
            // Stay - Lift - Hit
            if (t < wt.t_stay) {
                // Stay
                return wa.stay;
            } else if (t < wt.t_lift) {
                // Lift : stay -> lift (양끝 속도 0)
                if (wa.stay == wa.lift) {
                    return wa.stay;
                }
                return cubic_hermite(wt.t_stay, wa.stay, 0.0, wt.t_lift, wa.lift, 0.0, t);
            } else {
                // Hit : lift -> press (출발 속도 0), 2차
                if (wa.lift == wa.press) {
                    return wa.lift;
                }
                return quadratic(wt.t_lift, wa.lift, wt.t_hit, wa.press, true, t);
            }
        }

        case State::HIT_TO_HIT: {
            // Lift - Stay(lift) - Hit
            if (t < wt.t_stay) {
                // Lift : press -> lift (도착 속도 0), 2차
                if (wa.press == wa.lift) {
                    return wa.lift;
                }
                return quadratic(0.0, wa.press, wt.t_stay, wa.lift, false, t);
            } else if (t < wt.t_lift) {
                // Stay at lift angle
                return wa.lift;
            } else {
                // Hit : lift -> press (출발 속도 0), 2차
                if (wa.lift == wa.press) {
                    return wa.press;
                }
                return quadratic(wt.t_lift, wa.lift, wt.t_hit, wa.press, true, t);
            }
        }
    }

    return wa.stay;
}

double StateMotionGenerator::cubic_hermite(double ta, double qa, double va, double tb, double qb, double vb, double t) {
    // [ta,tb] 구간을 q(ta)=qa, q'(ta)=va, q(tb)=qb, q'(tb)=vb 로 보간하는 3차 다항식.
    double T = tb - ta;
    if (T == 0.0) {
        return qa;
    }
    double tau = (t - ta) / T;      // 0 ~ 1

    double h00 =  2.0*tau*tau*tau - 3.0*tau*tau + 1.0;
    double h10 =      tau*tau*tau - 2.0*tau*tau + tau;
    double h01 = -2.0*tau*tau*tau + 3.0*tau*tau;
    double h11 =      tau*tau*tau -     tau*tau;

    return h00*qa + h10*T*va + h01*qb + h11*T*vb;
}

double StateMotionGenerator::quadratic(double ta, double qa, double tb, double qb, bool zero_vel_at_start, double t) {
    // 두 위치(qa, qb)와 한쪽 끝 속도 0 으로 결정되는 2차 다항식.
    double T = tb - ta;
    if (T == 0.0) {
        return qa;
    }

    if (zero_vel_at_start) {
        // q(t) = qa + c2*(t-ta)^2,  q'(ta)=0,  qb 에서 c2 결정
        double dt = t - ta;
        double c2 = (qb - qa) / (T * T);
        return qa + c2 * dt * dt;
    } else {
        // q'(tb)=0 -> 정점이 tb. q(t) = qb + c2*(t-tb)^2, qa 에서 c2 결정
        double dt = t - tb;
        double c2 = (qa - qb) / (T * T);
        return qb + c2 * dt * dt;
    }
}