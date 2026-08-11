#include "common/command_queue.hpp"

CommandQueue::CommandQueue() {}

CommandQueue::~CommandQueue() {}

void CommandQueue::push(const ParsedCommand& cmd) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(cmd);
}

bool CommandQueue::empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

// ParsedCommand 타입 값이 있을수도 있고 없을 수도 있음
std::optional<ParsedCommand> CommandQueue::try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    ParsedCommand cmd = queue_.front();
    queue_.pop();
    return cmd;
}