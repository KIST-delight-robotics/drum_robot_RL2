#pragma once

#include <atomic>
#include <mutex>
#include <vector>
#include <string>

#include "common/play_session.hpp"

enum class RobotState { STANDBY, INIT, IDLE, PLAYING, SHUTTINGDOWN };

inline const char* state_to_string(RobotState s) {
    switch (s) {
        case RobotState::STANDBY:      return "STANDBY";
        case RobotState::INIT:         return "INIT";
        case RobotState::IDLE:         return "IDLE";
        case RobotState::PLAYING:      return "PLAYING";
        case RobotState::SHUTTINGDOWN: return "SHUTTINGDOWN";
        default:                       return "UNKNOWN";
    }
}

struct AppContext {
    std::atomic<bool> running{true};                // 전체 종료 플래그 (false 되면 모든 스레드 루프 탈출)
    std::atomic<bool> send_active{false};           // send_loop 활성화 신호
    std::atomic<bool> recv_active{false};           // recv_loop 활성화 신호

    std::atomic<RobotState> robot_state{RobotState::STANDBY};  // 로봇 상태

    std::mutex last_q_mutex;                        // 마지막 목표 관절각 스냅샷 (BehaviorPlanner가 갱신, TcpServer가 조회)
    std::vector<double> last_q_target_snapshot;     // NOTE: 실측값이 아니라 "마지막으로 명령된 목표 자세"임

    std::atomic<bool> play_abort{false};            // play 중 중단 플래그
    std::atomic<bool> pause_requested{false};       // abort 사유 구분 (true: PAUSE=재개 지점 저장 / false: stop·에러=폐기)
    std::atomic<double> play_speed_scale{1.0};      // play 중 속도 스케일

    std::mutex play_mutex;                          // 아래 두 필드 보호 (MotionPlanner가 갱신, 추후 TcpServer가 조회)
    std::string play_id;                            // 연주 중인 곡 id (빈 문자열 = 연주 아님)
    PausePoint pause_point;                         // 중단 시 저장, resume 시 소비
};