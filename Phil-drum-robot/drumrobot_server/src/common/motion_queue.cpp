#include "common/motion_queue.hpp"

MotionQueue::MotionQueue() {}

MotionQueue::~MotionQueue() {}

void MotionQueue::push(const MotionPrimitive& motion) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(motion);
}

bool MotionQueue::empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

// MotionPrimitive 타입 값이 있을수도 있고 없을 수도 있음
std::optional<MotionPrimitive> MotionQueue::try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    MotionPrimitive motion = queue_.front();
    queue_.pop();
    return motion;
}

// 맨 앞 원소를 꺼내지 않고 복사만 해서 반환 (소비 경로는 try_pop 하나로 유지)
std::optional<MotionPrimitive> MotionQueue::try_peek() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    return queue_.front();
}

void MotionQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<MotionPrimitive> empty;
    std::swap(queue_, empty);
}