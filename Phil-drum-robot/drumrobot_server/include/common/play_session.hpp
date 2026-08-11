#pragma once

#include <string>

// 연주 중단 시점의 재개 지점.
// 중단 시 MotionQueue에 남아 있던(=아직 궤적 생성 전인) 첫 이벤트의 마디 번호를 저장한다.
// play_id는 AppContext::play_id의 복사본이다. 연주가 완전히 끝나면(END 소진)
// AppContext::play_id는 비워지므로, 재개 대상 곡은 여기에 따로 들고 있어야 한다.
struct PausePoint {
    std::string play_id;    // 재개할 곡 id (play_list.json 기준)
    int bar = 0;            // 재개 시작 마디
    bool valid = false;     // 저장된 재개 지점이 있는지
};
