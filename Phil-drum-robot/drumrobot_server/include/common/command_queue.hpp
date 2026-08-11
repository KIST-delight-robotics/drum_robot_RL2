#pragma once

#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <optional>

// =============================================================
// 패킷 형식: OPCODE|arg1|arg2|...\n
// 필드 구분자: '|'   패킷 구분자: '\n'
//
// 예시:
//   LOOK|0|90\n               -> 정면 응시
//   GESTURE|nod\n             -> 끄덕임
//   MOVE|right_wrist|45|1.0\n -> right_wrist를 45도로 1.0초에 이동
//   POSE|home\n               -> home 포즈
//   QUIT\n                    -> 종료
// =============================================================

enum class Opcode {
    LOOK,       // 시선 제어       args: pan(deg), tilt(deg)
    GESTURE,    // 행동           args: type (nod / shake / wave / hi / hurray / happy)
    MOVE,       // 개별 관절 이동   args: motorName, angleDeg, [moveTime=2.0]
    POSE,       // 사전 정의 포즈   args: poseName (home / ready / shutdown)
    HIT,        // 드럼 타격       args: target (snare / ride / bass ...)
    PLAY,       // 악보 연주       args: scoreName
    PLAY_CTRL,  // 연주 제어       args: stop / speed
    REPLAY,     // CSV 궤적 오픈루프 재생 (Stage A)  args: csvPath

    // args 없음
    START,      // 시작
    READY,      // 상태 변경
    PAUSE,      // 연주 일시정지 (재개 지점 저장, PLAYING 전용)
    RESUME,     // 저장된 지점부터 연주 재개 (IDLE 전용)
    GET_STATUS, // 로봇 상태 조회 (응답: STATUS|<state>)
    QUIT,       // 종료
    UNKNOWN
};

struct ParsedCommand {
    bool valid = false;
    Opcode opcode = Opcode::UNKNOWN;
    std::vector<std::string> args;
};

class CommandQueue {
public:
    CommandQueue();
    ~CommandQueue();

    void push(const ParsedCommand& cmd);
    bool empty();
    std::optional<ParsedCommand> try_pop();

private:
    std::queue<ParsedCommand> queue_;
    std::mutex mutex_;
};