#include "trajectory/head_motion_generator.hpp"

HeadMotionGenerator::HeadMotionGenerator() {

}

HeadMotionGenerator::~HeadMotionGenerator() {

}

void HeadMotionGenerator::initialize(const std::map<int, InstrumentCoordinate>& coordinates) {
    drum_coordinates = coordinates;
}

HeadMotionPoint HeadMotionGenerator::reset(int note_r) {
    cur_note = note_r;

    prev_nod_intensity = 0.0;
    beat_sum = 0.0;

    HeadMotionPoint point;
    point.pitch = ready_angle;
    point.yaw = std::atan2(drum_coordinates[note_r].right_position[1], drum_coordinates[note_r].right_position[0]) - 90.0*M_PI/180.0;

    return point;
}

std::queue<HeadMotionPoint> HeadMotionGenerator::generate_motion(const std::vector<DrumEvent> rds, int num_point) {
    std::queue<HeadMotionPoint> out;
 
    if (rds.size() < 2 || num_point <= 0) {
        return out;
    }

    if (rds[1].note_num_R == 0) {
        next_note = cur_note;
    } else {
        next_note = rds[1].note_num_R;
    }

    // TODO: 오른손만 따라가는게 급하게 움직이거나 왼팔만 타격이 있을 때 어색해 보임 
    double cur_angle = (cur_note == 0) ? 0.0 : 
        std::atan2(drum_coordinates[cur_note].right_position[1], drum_coordinates[cur_note].right_position[0]);
    double next_angle = (next_note == 0) ? 0.0 : 
        std::atan2(drum_coordinates[next_note].right_position[1], drum_coordinates[next_note].right_position[0]);

    double nod_intensity = get_nod_intensity(rds);

    for (int i = 0; i < num_point; i++) {
        HeadMotionPoint point;

        double tau = (double)i / (num_point - 1.0);
        point.yaw = cur_angle + (next_angle - cur_angle) * (3.0 * std::pow(tau, 2.0) - 2.0 * std::pow(tau, 3.0)) - 90.0*M_PI/180.0;
        point.pitch = get_nod_angle(rds[1].beat, nod_intensity, i, num_point);

        out.push(point);
    }

    cur_note = next_note;

    return out;
}

double HeadMotionGenerator::get_nod_intensity(const std::vector<DrumEvent> rds) {
    int rds_size = (int)rds.size();

    double beat_sum = 0.0;
    int line = 0;
    double intensity_sum = 0.0;

    for (int i = 1; i < rds_size; i++) {
        // 계수 조절 가능
        double line_intensity = 0.1 * rds[i].velocity_R + 0.1 * rds[i].velocity_L + 0.5 * (rds[i].is_kick?1:0);
        intensity_sum += line_intensity;
        line++;

        beat_sum += rds[i].beat;        
        if (beat_sum >= 0.6) {  // 한 박
            break;
        }
    }

    return std::min(intensity_sum / line, 1.0);
}

float HeadMotionGenerator::get_nod_angle(double beat_of_line, double nod_intensity, int i, int n) {
    // 각도
    double nod_max = 20*M_PI/180.0;             // 고개가 움직이는 최대 각도
    double nod_angle = 0.0;

    // 세기
    // nodIntensity [0 1]
    double alpha = (i * nod_intensity + (n - i) * prev_nod_intensity) / (double)n;

    // beatSum 범위: [0 1)
    beat_sum += beat_of_line / 0.6 / n;
    beat_sum = beat_sum>=1.0?beat_sum-1.0:beat_sum;

    if (beat_sum < 0.7)
    {
        // 상승 궤적 생성
        double tau = beat_sum / 0.7;
        nod_angle = alpha * nod_max * (3.0 * pow(tau, 2) - 2.0 * pow(tau, 3));
    }
    else
    {   
        // 하강 궤적 생성
        double tau = (beat_sum - 0.7) / 0.3;
        nod_angle = alpha * nod_max * (1.0 - (3.0 * pow(tau, 2) - 2.0 * pow(tau, 3)));
    }

    if (i == n - 1)
    {
        prev_nod_intensity = nod_intensity;
    }

    // 고개를 아래로 숙이는 방향이 (+)
    return ready_angle - nod_angle;
}