#pragma once
#include "tensor.h"

class Linear {
public:
    //random weights
    Linear(size_t in_features, size_t out_features, bool bias = true);

    //existent weight, for that reason we pass weight as a parameters
    Linear(std::vector<float> weight, std::vector<float> bias,
           size_t in_features, size_t out_features);

    Tensor forward(const Tensor& x) const;

    const Tensor& weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }

private:
    Tensor weight_;
    bool has_bias_;
    Tensor bias_;
};
