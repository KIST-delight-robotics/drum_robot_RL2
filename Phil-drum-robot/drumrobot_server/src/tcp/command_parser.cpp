#include "tcp/command_parser.hpp"

CommandParser::CommandParser() {

}

CommandParser::~CommandParser() {

}

ParsedCommand CommandParser::parse(const std::string& cmd) {
    ParsedCommand result;

    // 앞뒤 공백·개행 제거
    std::string cleaned = trim(cmd);
    if (cleaned.empty()) {
        std::cerr << "[CommandParser] Empty command\n";
        return result;
    }

    // '|' 구분자로 분리
    std::vector<std::string> tokens = split(cleaned, '|');
    if (tokens.empty()) {
        std::cerr << "[CommandParser] No tokens found: " << cleaned << "\n";
        return result;
    }

    // 첫 번째 토큰 -> Opcode
    Opcode opcode = to_opcode(tokens[0]);
    if (opcode == Opcode::UNKNOWN) {
        std::cerr << "[CommandParser] Unknown opcode: " << tokens[0] << "\n";
        result.opcode = Opcode::UNKNOWN;
        return result;
    }

    // 나머지 토큰 -> args
    std::vector<std::string> args(tokens.begin() + 1, tokens.end());

    // 인자 수 검증
    if (!validate_args(opcode, args)) {
        std::cerr << "[CommandParser] Invalid args for opcode: " << tokens[0] << "\n";
        return result;
    }

    result.valid  = true;
    result.opcode = opcode;
    result.args   = args;
    return result;
}

std::vector<std::string> CommandParser::split(const std::string& str, char delimiter) const {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream stream(str);
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

std::string CommandParser::trim(const std::string& str) const {
    const std::string whitespace = " \t\r\n";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

Opcode CommandParser::to_opcode(const std::string& token) const {
    // 대소문자 무시
    std::string upper = token;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    if (upper == "LOOK")    return Opcode::LOOK;
    if (upper == "GESTURE") return Opcode::GESTURE;
    if (upper == "MOVE")    return Opcode::MOVE;
    if (upper == "POSE")    return Opcode::POSE;
    if (upper == "HIT")     return Opcode::HIT;
    if (upper == "PLAY")    return Opcode::PLAY;
    if (upper == "START")   return Opcode::START;
    if (upper == "READY")   return Opcode::READY;
    if (upper == "PAUSE")   return Opcode::PAUSE;
    if (upper == "RESUME")  return Opcode::RESUME;
    if (upper == "PLAY_CTRL") return Opcode::PLAY_CTRL;
    if (upper == "REPLAY")  return Opcode::REPLAY;
    if (upper == "QUIT" || upper == "Q") return Opcode::QUIT;

    return Opcode::UNKNOWN;
}

bool CommandParser::validate_args(Opcode opcode, const std::vector<std::string>& args) const {
    switch (opcode) {
        case Opcode::LOOK:      return args.size() >= 2;    // pan, tilt
        case Opcode::GESTURE:   return args.size() >= 1;    // type
        case Opcode::MOVE:      return args.size() >= 2;    // motorName, angleDeg, [moveTime]
        case Opcode::POSE:      return args.size() >= 1;    // poseName
        case Opcode::HIT:       return args.size() >= 1;    // target
        case Opcode::PLAY:      return args.size() >= 1;    // scoreName
        case Opcode::PLAY_CTRL: return args.size() >= 1;    // stop / speed
        case Opcode::REPLAY:    return args.size() >= 1;    // csvPath
        case Opcode::START:     return true;                // 인자 없음
        case Opcode::READY:     return true;                // state
        case Opcode::PAUSE:     return true;                // 인자 없음
        case Opcode::RESUME:    return true;                // 인자 없음
        case Opcode::QUIT:      return true;                // 인자 없음
        default:                return false;
    }
}