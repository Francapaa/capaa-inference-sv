#include "server_queue.h"

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

size_t ServerQueue::size() const{

}

bool ServerQueue::empty() const{

}