#pragma once
#include "tensor.h"

namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b);

Tensor add(const Tensor& a, const Tensor& b);

Tensor mul(const Tensor& t, float scalar);

Tensor softmax(const Tensor& t);

Tensor layer_norm(const Tensor& t, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);

Tensor silu(const Tensor& t);

} // namespace ops of tensors
