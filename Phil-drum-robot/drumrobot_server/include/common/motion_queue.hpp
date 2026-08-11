#pragma once

#include <queue>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

enum class MotionType { STANDBY, TRANSLATE, DRUM, IDLE, REPLAY };
enum class TrajectorySpace { JOINT, TASK };
enum class TrajectoryProfile { TRAPEZOIDAL, CUBIC, QUINTIC, COSINE };

enum class PlayFlag { START, PLAYING, END };
struct DrumEvent {
    unsigned int bar = 1;           // 마디 번호
    double beat = 0.6;              // 박자 (0.6 -> 한 박)
    int note_num_R = 0;             // 오른팔 타격 악기 번호
    int note_num_L = 0;             // 왼팔 타격 악기 번호
    int velocity_R = 0;             // 오른팔 타격 강도
    int velocity_L = 0;             // 왼팔 타격 강도
    bool is_kick = false;           // bass drum
    bool is_closed_hihat = false;   // closed hi-hat
    double t = 0.0;                 // 누적 시간
};

struct MotionPrimitive {
    MotionType type = MotionType::TRANSLATE;

    // STANDBY
    // 키 제거하기 전 현재 위치 유지

    // TRANSLATE
    TrajectorySpace space = TrajectorySpace::JOINT;
    TrajectoryProfile profile = TrajectoryProfile::COSINE;
    std::vector<double> q_target;       // joint space일 때
    std::vector<double> p_target_R;     // task space일 때
    std::vector<double> p_target_L;
    double t_total = 4.0;

    // DRUM
    std::vector<DrumEvent> robotic_drum_score;
    PlayFlag flag = PlayFlag::PLAYING;
    int init_note_r = 1, init_note_l = 1;    // 초기 위치 기본값: 스네어

    // IDLE
    // ... 별도 필드

    // REPLAY (Stage A: sim에서 뽑은 CSV 관절궤적 오픈루프 재생)
    std::string replay_csv;   // CSV 파일 경로
};

class MotionQueue {
public:
    MotionQueue();
    ~MotionQueue();

    void push(const MotionPrimitive& motion);
    bool empty();
    std::optional<MotionPrimitive> try_pop();
    std::optional<MotionPrimitive> try_peek();
    void clear();

private:
    std::queue<MotionPrimitive> queue_;
    std::mutex mutex_;
};