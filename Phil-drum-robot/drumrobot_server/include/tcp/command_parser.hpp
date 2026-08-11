#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>

#include "common/command_queue.hpp"

class CommandParser {
public:
    CommandParser();
    ~CommandParser();

    ParsedCommand parse(const std::string& cmd);
 
private:
    // '|' 구분자로 토큰 분리
    std::vector<std::string> split(const std::string& str, char delimiter) const;
 
    // 앞뒤 공백 및 개행 제거
    std::string trim(const std::string& str) const;
 
    // 문자열 → Opcode 변환
    Opcode to_opcode(const std::string& token) const;
 
    // Opcode별 최소 인자 수 검증
    bool validate_args(Opcode opcode, const std::vector<std::string>& args) const;
};