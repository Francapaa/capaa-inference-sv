#pragma once
#include "tensor.hpp"
#include <vector>
#include <random>

class BigramModel {
public:
    explicit BigramModel(size_t vocab_size); // explicit bcs receive only vocab_size

    void train(const std::vector<int>& tokens);
    void build();

    const float* forward(int token_id) const;
    int sample(const float* logits_row, float temperature) const;
    std::vector<int> generate(const std::vector<int>& prompt,
                              int max_tokens,
                              float temperature) const; // generate the next token

    size_t vocab_size() const { return vocab_size_; } // getter

private:
    size_t vocab_size_;
    Tensor counts_;
    Tensor logits_; // score = puntaje
    bool built_ = false;

    mutable std::mt19937 rng_{42};
};
