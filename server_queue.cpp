#include <iostream>
#include <chrono>
#include "serverHTTP.cpp"
#include <queue>

struct server_queue {
private:
	int id = 0;
	bool running = false;
	bool sleeping = false;

    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task&&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;

public:
    int post(server_task&& task, bool front = false);
    int get_new_id();
};


int server_queue::post(server_task && task, bool front) {

	std::queue<InferenceRequest> request_queue; 

    const int id = task.id;

    std::cout << "New task with ID:" << id << std::endl;

    if (front) {
        queue_tasks.push_front(task);
    }
    else {
        queue_tasks.push_back(task); 
    }

    condition_tasks.notify_one(); 

    return id_tasks;
}

int server_queue::get_new_id() {
    std::unique_lock::<std::mutex> lock(mutex_task);
    return id + 1; // is not necessary to declare the variable
}