#pragma once
#include <deque>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
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

    mutable std::mutex mutex_tasks_;
    std::condition_variable condition_tasks_;

    std::function<void(InferenceRequest&&)> callback_new_task_;
    std::function<void()> callback_update_slots_;
    std::function<void(bool)> callback_sleeping_state_;

public:
    uint64_t post(InferenceRequest task, bool front = false);
    size_t size() const;
    bool empty() const;
};
