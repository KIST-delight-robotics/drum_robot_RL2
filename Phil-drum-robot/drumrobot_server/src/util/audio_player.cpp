#include "util/audio_player.hpp"

#include <iostream>

#include "miniaudio/miniaudio.h"

AudioPlayer::AudioPlayer() {}

AudioPlayer::~AudioPlayer() {
    if (engine_) {
        ma_engine_uninit(engine_);
        delete engine_;
        engine_ = nullptr;
    }
    if (context_) {
        ma_context_uninit(context_);
        delete context_;
        context_ = nullptr;
    }
}

bool AudioPlayer::initialize() {
    if (ready_) return true;

    context_ = new ma_context();
    engine_  = new ma_engine();

    // PulseAudio 백엔드만 사용
    ma_backend backends[] = { ma_backend_pulseaudio };
    ma_context_config cfg = ma_context_config_init();

    if (ma_context_init(backends, 1, &cfg, context_) != MA_SUCCESS) {
        std::cerr << "[AudioPlayer] 컨텍스트 초기화 실패\n";
        delete context_; context_ = nullptr;
        delete engine_;  engine_  = nullptr;
        return false;
    }

    ma_engine_config ecfg = ma_engine_config_init();
    ecfg.pContext = context_;
    if (ma_engine_init(&ecfg, engine_) != MA_SUCCESS) {
        std::cerr << "[AudioPlayer] 엔진 초기화 실패\n";
        ma_context_uninit(context_);
        delete context_; context_ = nullptr;
        delete engine_;  engine_  = nullptr;
        return false;
    }

    std::cout << "[AudioPlayer] 백엔드: " << ma_get_backend_name(context_->backend) << "\n";
    ready_ = true;
    return true;
}

void AudioPlayer::set_track(const std::string& name) {
    track_ = name;
}

void AudioPlayer::clear_track() {
    track_.clear();
}

std::string AudioPlayer::make_path(const std::string& name) const {
    return base_dir_ + "/" + name + ".wav";
}

bool AudioPlayer::play() {
    if (!ready_) {
        std::cerr << "[AudioPlayer] 엔진 미초기화\n";
        return false;
    }
    if (track_.empty()) {
        // 곡 미설정(솔로 등) = 정상적으로 아무것도 재생하지 않음
        return false;
    }

    const std::string path = make_path(track_);
    if (ma_engine_play_sound(engine_, path.c_str(), nullptr) != MA_SUCCESS) {
        std::cerr << "[AudioPlayer] 재생 실패: " << path << "\n";
        return false;
    }
    return true;
}

bool AudioPlayer::play(const std::string& name) {
    set_track(name);
    return play();
}

void AudioPlayer::stop() {
    if (ready_ && engine_) {
        ma_engine_stop(engine_);
    }
}