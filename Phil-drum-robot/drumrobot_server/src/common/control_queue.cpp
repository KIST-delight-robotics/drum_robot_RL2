#include "common/control_queue.hpp"

ControlQueue::ControlQueue() {}

ControlQueue::~ControlQueue() {}

void ControlQueue::push(const ControlSetPoint& point) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(point);
}

bool ControlQueue::empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t ControlQueue::size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

// ControlSetPoint 타입 값이 있을수도 있고 없을 수도 있음
std::optional<ControlSetPoint> ControlQueue::try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    ControlSetPoint point = queue_.front();
    queue_.pop();
    return point;
}