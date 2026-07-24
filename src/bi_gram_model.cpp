#include "include/bi_gram_model.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

BigramModel::BigramModel(size_t vocab_size)
    : vocab_size_(vocab_size)
    , counts_({vocab_size, vocab_size})
    , logits_({vocab_size, vocab_size})
{}

void BigramModel::train(const std::vector<int>& tokens) {
    if (tokens.size() < 2) return; // we need at least 2 tokens

    /*
        we add 1.0f in every 2 tokens
        tokens = [12, 45, 78, 3]
        (12,45) counts_[12, 45] + 1
        (78,3) counts_[78, 3] + 1
        after we train our neural net, we have the number that the token j was exactly after 
        the token i
    */ 
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        counts_.at({static_cast<size_t>(tokens[i]),
                    static_cast<size_t>(tokens[i + 1])}) += 1.0f;
    }
}

void BigramModel::build() {
    /*
    
        contadores -> probab-logs each row 
        we take counts_[i], add all the counter of its own row
        if (fila has data) -> / each cell by the sum P(j|i) , apply log() and save in logits()
        if (fila hasnt data) -> (sum = 0) assign uniform distribution log(1/V) to all the row

        if counts_[0] = [2, 1, 0, 1] (V=4, sum=4):
        P(0|0) = 2/4 = 0.5 → log(0.5) = -0.693
        P(1|0) = 1/4 = 0.25 → log(0.25) = -1.386
        P(2|0) = 0/4 = 0 → log(0 + 1e-10) = -23.0  
        P(3|0) = 1/4 = 0.25 → log(0.25) = -1.386
    */


    for (size_t i = 0; i < vocab_size_; ++i) {
        float* row_counts = counts_.data() + i * vocab_size_;
        float* row_logits = logits_.data() + i * vocab_size_;

        float sum = 0.0f;
        for (size_t j = 0; j < vocab_size_; ++j) {
            sum += row_counts[j];
        }

        if (sum > 0.0f) {
            float inv_sum = 1.0f / sum;
            for (size_t j = 0; j < vocab_size_; ++j) {
                row_logits[j] = std::log(row_counts[j] * inv_sum + 1e-10f);
            }
        } else {
            float uniform_log = std::log(1.0f / static_cast<float>(vocab_size_));
            for (size_t j = 0; j < vocab_size_; ++j) {
                row_logits[j] = uniform_log;
            }
        }
    }
    built_ = true;
}

const float* BigramModel::forward(int token_id) const {
    /*
        return a pointer to the row token_id of logits_. That row contains "v" floats,
        which are prob-log of each next token
    */
    if (token_id >= vocab_size()) {
        throw std::runtime_error("The token cannot be greater than the vocab_size")
    }

    return logits_.data() + static_cast<size_t>(token_id) * vocab_size_;
}

int BigramModel::sample(const float* logits_row, float temperature) const {

    /*
        if (temperature) 
        scale logits by 1/temp
        substract max_val to numeric estability 
        exp() -> prob
        normalize dividing by sum
        accumulates (cumsum) and search the first index where random [0,1) < cum
    */
    size_t n = vocab_size_;

    if (temperature < 1e-6f) {
        return static_cast<int>(
            std::max_element(logits_row, logits_row + n) - logits_row);
    }

    std::vector<float> probs(n);
    float inv_temp = 1.0f / temperature;

    float max_val = *std::max_element(logits_row, logits_row + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        probs[i] = std::exp((logits_row[i] - max_val) * inv_temp);
        sum += probs[i];
    }

    float r = std::uniform_real_distribution<float>{0.0f, 1.0f}(rng_);
    float cum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cum += probs[i] / sum;
        if (r < cum) return static_cast<int>(i);
    }

    return static_cast<int>(n - 1);
}

std::vector<int> BigramModel::generate(const std::vector<int>& prompt,
                                       int max_tokens,
                                       float temperature) const {

    /*
        take the last token of the prompt as a seed
        loop to max_tokens:
        1) forward(seed) -> get the row of prob-log to the current token
        2) sample() -> choose the next token with temperature and add to output
        3) the new token becomes to the new seed
        EXAMPLE:
        prompt [12, 45, 78], seed = 78:
        seed=78 → forward(78) → sample → 3  → output=[3],  seed=3
        seed=3  → forward(3)  → sample → 99 → output=[3,99],  seed=99
        seed=99 → forward(99) → sample → 4  → output=[3,99,4], seed=4
    */

    if (!built_) {
        throw std::runtime_error(
            "BigramModel not built, call build() first");
    }
    if (prompt.empty()) {
        throw std::invalid_argument("prompt cannot be empty");
    }
    // some validations


    std::vector<int> output;
    output.reserve(static_cast<size_t>(max_tokens));

    int seed = prompt.back();
    for (int i = 0; i < max_tokens; ++i) {
        const float* row = forward(seed);
        int next = sample(row, temperature);
        output.push_back(next);
        seed = next;
    }

    return output;
}
