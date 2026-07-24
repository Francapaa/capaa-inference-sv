#include "include/layers.hpp"
#include "include/ops.hpp"
#include <cmath>

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : weight_(std::vector<size_t>{in_features, out_features})
    , has_bias_(bias)
    , bias_(has_bias_ ? Tensor(std::vector<size_t>{out_features}) : Tensor{})
{
    // He init: uniform random in [-sqrt(6/in), sqrt(6/in)]
    float scale = std::sqrt(6.0f / static_cast<float>(in_features));
    for (size_t i = 0; i < weight_.size(); ++i) {
        weight_[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f * scale - scale;
    }
    if (has_bias_) {
        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] = 0.0f;
        }
    }
}

Linear::Linear(std::vector<float> weight, std::vector<float> bias,
               size_t in_features, size_t out_features)
    : weight_({in_features, out_features}, std::move(weight))
    , has_bias_(!bias.empty())
    , bias_(has_bias_ ? Tensor({out_features}, std::move(bias)) : Tensor{})
{}

Tensor Linear::forward(const Tensor& x) const {
    // x: [batch, in_features]
    // weight: [in_features, out_features]
    // result = x @ weight + bias
    auto result = ops::matmul(x, weight_);

    if (has_bias_) {
        // add bias to each row
        size_t batch = result.shape()[0];
        size_t out = result.shape()[1];
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < out; ++j) {
                result.at({i, j}) += bias_[j];
            }
        }
    }
    return result;
}
