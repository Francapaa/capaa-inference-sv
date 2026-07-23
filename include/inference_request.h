#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <optional>

struct InferenceRequest {
    uint64_t id;
    std::string prompt;
    std::vector<int> tokens;

    float temperature = 0.7f;
    int max_quantity_of_tokens = 128;

    bool is_finished = false;
    std::vector<int> output_tokens;
    std::string error_message; 

    std::chrono::steady_clock::time_point enqueue_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
};
