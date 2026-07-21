#include <iostream>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "inference_request.h"

class ServerQueue {
private:
    uint64_t next_id_ = 0;

    bool running_ = false;
    bool sleeping_ = false;
    bool req_stop_sleeping_ = false;
    int64_t time_last_task_ = 0;

    std::deque<InferenceRequest> queue_tasks_;
    std::deque<InferenceRequest> queue_tasks_deferred_;

    std::mutex mutex_tasks_;
    std::condition_variable condition_tasks_;

    std::function<void(InferenceRequest&&)> callback_new_task_;
    std::function<void()> callback_update_slots_;
    std::function<void(bool)> callback_sleeping_state_;

public:
    uint64_t post(InferenceRequest task, bool front = false);
};

uint64_t ServerQueue::post(InferenceRequest task, bool front) {
    {
        std::lock_guard<std::mutex> lock(mutex_tasks_);
        task.id = next_id_++;
        task.enqueue_time = std::chrono::steady_clock::now();
        if (front) {
            queue_tasks_.push_front(std::move(task));
        } else {
            queue_tasks_.push_back(std::move(task));
        }
    }
    condition_tasks_.notify_one();
    return next_id_ - 1;
}