#pragma once

#include <string>

// miniaudio 전방 선언용 타입 (헤더에 miniaudio.h를 노출하지 않기 위함)
struct ma_engine;
struct ma_context;

// 음악 재생 유틸.
// 사용법:
//   AudioPlayer audio;
//   audio.initialize();                       // PulseAudio 엔진 1회 초기화
//   audio.set_track("TIM");                   // 곡명(확장자 제외) 미리 설정
//   audio.play();                             // 설정된 곡을 즉시 재생 (fire-and-forget)
//   audio.clear_track();                      // 연주 종료 후 곡 초기화
//
// 곡명만 받으면 내부에서 "<base_dir>/<name>.wav" 경로로 변환한다.
// set_track 으로 미리 곡을 지정해두면, 실시간 송신 루프(send_loop)에서는
// play() 한 번만 호출하면 되므로 타격 시작 순간에 곧바로 소리를 낼 수 있다.
//
// 주의: set_track 은 "이번 연주 한정 1회용 설정"이다. 연주가 끝나거나 중단되는
//       모든 경로에서 clear_track() 으로 반드시 리셋해야, 다음에 배경음악 없는
//       솔로 연주 시 이전 곡이 다시 재생되지 않는다.
class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();

    AudioPlayer(const AudioPlayer&) = delete;
    AudioPlayer& operator=(const AudioPlayer&) = delete;

    // PulseAudio 백엔드로 엔진 초기화. 성공 시 true.
    bool initialize();

    // 재생할 곡명(확장자 제외)을 미리 설정. 예: "TIM" -> data/audio/TIM.wav
    void set_track(const std::string& name);

    // 설정된 곡명 초기화. 연주 종료/중단 후 호출해 다음 솔로에서 잔여 곡이
    // 재생되지 않게 한다.
    void clear_track();

    // 미리 설정된 곡을 즉시 재생. 곡이 설정돼 있지 않으면(솔로 등) 아무것도
    // 재생하지 않고 false 를 반환한다. (정상 동작이며 에러가 아님)
    bool play();

    // 곡명을 즉석에서 지정하면서 바로 재생.
    bool play(const std::string& name);

    // 재생 중인 모든 소리 정지.
    void stop();

    bool is_ready() const { return ready_; }

private:
    std::string make_path(const std::string& name) const;

    ma_context* context_ = nullptr;
    ma_engine*  engine_  = nullptr;
    bool ready_ = false;

    std::string base_dir_ = "drumrobot_server/data/audio";
    std::string track_;     // 미리 설정된 곡명 (확장자 제외)
};