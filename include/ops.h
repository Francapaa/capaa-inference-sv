#pragma once
#include "tensor.h"

namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b);

Tensor add(const Tensor& a, const Tensor& b);

Tensor mul(const Tensor& t, float scalar);

Tensor softmax(const Tensor& t);

Tensor RMSNorm(const Tensor& t, const Tensor& gamma, float eps = 1e-5f);

Tensor silu(const Tensor& t);

} // namespace ops of tensors
