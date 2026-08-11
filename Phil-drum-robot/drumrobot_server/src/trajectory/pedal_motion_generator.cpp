#include "trajectory/pedal_motion_generator.hpp"

PedalMotionGenerator::PedalMotionGenerator() {

}

PedalMotionGenerator::~PedalMotionGenerator() {

}

PedalMotionPoint PedalMotionGenerator::reset() {
    PedalMotionPoint point;
    point.right = ready_angle;
    point.left = open_hihat_angle;

    return point;
}

std::queue<PedalMotionPoint> PedalMotionGenerator::generate_motion(const std::vector<DrumEvent> rds, int num_point, double dt) {
    std::queue<PedalMotionPoint> out;
 
    if (rds.size() < 2 || num_point <= 0) {
        return out;
    }

    State bass_state = get_state(rds[0].is_kick, rds[1].is_kick);
    State hihat_state = get_state(rds[0].is_closed_hihat, rds[1].is_closed_hihat);

    BassParam bass_param = get_bass_param(rds[0].t, rds[1].t);
    HihatParam hihat_param = get_hihat_param(rds[0].t, rds[1].t);

    for (int i = 0; i < num_point; i++) {
        PedalMotionPoint point;

        point.right = get_bass_angle(i * dt, bass_param, bass_state);
        point.left = get_hihat_angle(i * dt, hihat_param, hihat_state);

        out.push(point);
    }

    return out;
}

PedalMotionGenerator::State PedalMotionGenerator::get_state(bool cur, bool next) {
    if (!cur && !next)     return State::REST_TO_REST;
    else if (cur && !next) return State::HIT_TO_REST;
    else if (!cur && next) return State::REST_TO_HIT;
    else return State::HIT_TO_HIT;  // cur && next
}

PedalMotionGenerator::BassParam PedalMotionGenerator::get_bass_param(double t0, double t1) {
    BassParam bass_param;

    double T = t1 - t0;
    bass_param.t_lift = std::max(0.6 * (T), T - 0.2);   // 최고점 시간, 전체 타격 시간의 60% , 타격 시간의 최대값은 0.2초
    bass_param.t_stay = 0.45 * (T);                     // 복귀 시간, 전체 타격 시간의 45%
    bass_param.t_hit = T;                               // 전체 타격 시간

    return bass_param;
}

PedalMotionGenerator::HihatParam PedalMotionGenerator::get_hihat_param(double t0, double t1) {
    HihatParam hihat_param;

    double T = t1 - t0;
    hihat_param.t_lift = 0.1 * T;
    hihat_param.t_settling = 0.9 * T;
    hihat_param.t_hit = T;
    hihat_param.t_splash = std::max(0.5*T, T - 0.1);    // 0.1초에 20도 이동이 안전. 그 이하는 모터 멈출 수 있음.

    return hihat_param;
}

double PedalMotionGenerator::get_bass_angle(double t, BassParam bp, State bs) {
    // Xl : lift 궤적 , Xh : hit 궤적
    double Xl = 0.0, Xh = 0.0;

    // 타격 시간이 0.2초 이하일 때
    if (bp.t_hit <= 0.2) {
        if (bs == State::HIT_TO_HIT) {
            // 짧은 시간에 연속으로 타격하는 경우 덜 들도록, 올라오는 궤적과 내려가는 궤적을 합쳐서 사용 (- 영역이라 가능)
            // 조정수 박사님 자료 BassAngle 부분 참고
            double temp_liftTime = bp.t_hit / 2;
            Xl = -0.5 * (press_angle - ready_angle) * (cos(M_PI * (t / bp.t_hit + 1)) - 1);

            if (t < temp_liftTime) {
                Xh = 0;
            } else {
                Xh = -0.5 * (press_angle - ready_angle) * (cos(M_PI * (t - temp_liftTime) / (bp.t_hit - temp_liftTime)) - 1);
            }
        } else if (bs == State::HIT_TO_REST) {
            // 시간이 짧을 땐 전체 시간을 다 써서 궤적 생성
            // 복귀하는 궤적
            Xl = -0.5 * (press_angle - ready_angle) * (cos(M_PI * (t / bp.t_hit + 1)) - 1);
            Xh = 0;
        } else if (bs == State::REST_TO_HIT) {
            // 타격하는 궤적
            Xl = 0;
            Xh = -0.5 * (press_angle - ready_angle) * (cos(M_PI * t / bp.t_hit) - 1);
        } else {
            Xl = 0;
            Xh = 0;
        } 
    } else if (t < bp.t_stay) {
        // 0.2초 이상일 때 
        // StayTime 이전 -> 타격 후 들어올림   
        if(bs == State::HIT_TO_REST || bs == State::HIT_TO_HIT) {
            // 복귀하는 궤적
            Xl = -0.5 * (press_angle - ready_angle) * (cos(M_PI * (t / bp.t_stay + 1)) - 1);
            Xh = 0;
        } else {
            Xl = 0;
            Xh = 0;
        }
    } else if (t > bp.t_lift && t <= bp.t_hit) {
        // LiftTime부터 HitTime까지 -> 타격
        if (bs == State::REST_TO_HIT || bs == State::HIT_TO_HIT) {
            // 타격하는 궤적
            Xl = 0;
            Xh = -0.5 * (press_angle - ready_angle) * (cos(M_PI * (t - bp.t_lift) / (bp.t_hit - bp.t_lift)) - 1);
        } else {
            Xl = 0;
            Xh = 0;
        }
    }
    
    return ready_angle + Xl + Xh;        // 준비 각도 + 하강 각도 + 상승 각도
}

double PedalMotionGenerator::get_hihat_angle(double t, HihatParam hp, State hs, bool is_splash) {
    // 1.open/closed, 2.splash 두 개의 하이햇 연주법을 구현함.
    // o/c 상태에 따라 악기 소리가 다름. 소리나는 도중에 상태가 변해도 소리가 변함. 
    // 정확한 소리를 내기 위해 타격 이전(settlingTime)에 상태를 바꾸고 타격 이후(liftTime)에도 그 상태를 유지함.
    // splash는 페달을 밟았다가 떼며 두 심벌이 부딪힘으로 소리내는 방식임. -> 현재 사용 안함
    double hihat_angle = 0.0;

    if(hs == State::REST_TO_REST) {
        hihat_angle = open_hihat_angle;
    } else if(hp.t_hit <= 0.2) {
        // 한 박의 시간이 0.2초 이하인 경우
        if(hs == State::HIT_TO_REST) {
            hihat_angle = cosine_profile(closed_hihat_angle, open_hihat_angle, 0, hp.t_hit, t);       // 전체 시간동안 궤적 생성
        } else if(is_splash) {
            // Hihat splash 궤적
            if(t < hp.t_splash) {
                if(hs == State::REST_TO_HIT) {
                    hihat_angle = cosine_profile(open_hihat_angle, closed_hihat_angle, 0, hp.t_hit, t);
                } else if(hs == State::HIT_TO_HIT) {
                    hihat_angle = cosine_profile(closed_hihat_angle, closed_hihat_angle - (closed_hihat_angle - open_hihat_angle) * hp.t_hit / 0.2, 0, hp.t_splash, t);     // 20도/0.1초 의 속도를 유지하기 위해서 시간에 따라 이동량 감소시킴.
                }
            } else {
                if(hs == State::REST_TO_HIT) {
                    hihat_angle = cosine_profile(open_hihat_angle, closed_hihat_angle, 0, hp.t_hit, t);
                } else if(hs == State::HIT_TO_HIT) {
                    hihat_angle = cosine_profile(closed_hihat_angle - (closed_hihat_angle - open_hihat_angle) * hp.t_hit / 0.2, closed_hihat_angle, hp.t_splash, hp.t_hit, t);
                }
            }
        } else {
            // open/closed Hihat 궤적
            if(hs == State::REST_TO_HIT) {
                hihat_angle = cosine_profile(open_hihat_angle, closed_hihat_angle, 0, hp.t_hit, t);
            } else if(hs == State::HIT_TO_HIT) {                
                hihat_angle = closed_hihat_angle;
            }
        }
    } else if(is_splash) {
        // 한 박의 시간이 0.2초 초과인 경우
        // Hi-Hat splash 궤적
        if(t < hp.t_splash) {
            if(hs == State::REST_TO_HIT) {
                hihat_angle = open_hihat_angle;
            } else if(hs == State::HIT_TO_HIT) {
                hihat_angle = cosine_profile(closed_hihat_angle, open_hihat_angle, 0, hp.t_splash, t);
            }
        } else if(t >= hp.t_splash) {
            if(hs == State::REST_TO_HIT) {
                hihat_angle = cosine_profile(open_hihat_angle, closed_hihat_angle, hp.t_splash, hp.t_hit, t);
            } else if(hs == State::HIT_TO_HIT) {
                hihat_angle = cosine_profile(open_hihat_angle, closed_hihat_angle, hp.t_splash, hp.t_hit, t);
            }
        }
    } else {
        // open/closed Hi-Hat 궤적
        if(t < hp.t_lift) {
            if(hs == State::HIT_TO_REST) {
                hihat_angle = closed_hihat_angle;
            } else if(hs == State::REST_TO_HIT) {
                hihat_angle = open_hihat_angle;
            } else if(hs == State::HIT_TO_HIT) {                
                hihat_angle = closed_hihat_angle;
            }
        } else if(t >= hp.t_lift && t < hp.t_settling) {
            if(hs == State::HIT_TO_REST) {
                hihat_angle = cosine_profile(closed_hihat_angle, open_hihat_angle, hp.t_lift, hp.t_settling, t);
            } else if(hs == State::REST_TO_HIT) {      
                hihat_angle = cosine_profile(open_hihat_angle, closed_hihat_angle, hp.t_lift, hp.t_settling, t);
            } else if(hs == State::HIT_TO_HIT) {
                hihat_angle = closed_hihat_angle;
            }
        } else if(t >= hp.t_settling) {
            if(hs == State::HIT_TO_REST) {
                hihat_angle = open_hihat_angle;
            } else if(hs == State::REST_TO_HIT) {
                hihat_angle = closed_hihat_angle;
            } else if(hs == State::HIT_TO_HIT) {
                hihat_angle = closed_hihat_angle;
            }
        }
    }

    return hihat_angle;
}

double PedalMotionGenerator::cosine_profile(double qi, double qf, double ti, double tf, double t) {
    // ti <= t <= tf
    double A, B, w;

    A = (qi - qf) / 2;
    B = (qi + qf) / 2;
    w = M_PI / (tf - ti);

    return A * std::cos(w*(t - ti)) + B;
}