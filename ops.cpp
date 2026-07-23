#include "ops.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b) { // matrix multiplication 
    auto shape_a = a.shape();
    auto shape_b = b.shape();
    if (shape_a.size() != 2 || shape_b.size() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }
    size_t m = shape_a[0], k = shape_a[1], n = shape_b[1];
    if (shape_b[0] != k) {
        throw std::invalid_argument("matmul dimension mismatch");
    }

    Tensor result({m, n});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t t = 0; t < k; ++t) {
                sum += a.at({i, t}) * b.at({t, j});
            }
            result.at({i, j}) = sum;
        }
    }
    return result;
}

Tensor add(const Tensor& a, const Tensor& b) { // matrix add
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("add requires same shape");
    }
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Tensor mul(const Tensor& t, float scalar) { // tensor multiplication
    Tensor result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) {
        result[i] = t[i] * scalar;
    }
    return result;
}

static void softmax_inplace(float* data, size_t n) { 
    float max_val = *std::max_element(data, data + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (size_t i = 0; i < n; ++i) {
        data[i] /= sum;
    }
}

Tensor softmax(const Tensor& t) {
    auto shape = t.shape();
    if (shape.empty()) throw std::invalid_argument("softmax on empty tensor");

    size_t last_dim = shape.back();
    size_t outer = t.size() / last_dim;

    Tensor result(shape);
    for (size_t i = 0; i < outer; ++i) {
        float* slice = result.data() + i * last_dim;
        std::copy(t.data() + i * last_dim, t.data() + (i + 1) * last_dim, slice);
        softmax_inplace(slice, last_dim);
    }
    return result;
}

static void rms_norm_inplace(float* data, const float* gamma, size_t n, float eps) {
    float sum_sq = 0.0f; // sum square root
    for (size_t i = 0; i < n; ++i) {
        sum_sq += data[i] * data[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(n) + eps);
    float inv_rms = 1.0f / rms;
    for (size_t i = 0; i < n; ++i) {
        data[i] = (data[i] * inv_rms) * gamma[i];
    }
}

Tensor RMSNorm(const Tensor& t, const Tensor& gamma, float eps) {
    auto shape = t.shape();
    if (shape.empty()) throw std::invalid_argument("RMSNorm on empty tensor");

    size_t last_dim = shape.back();
    if (gamma.size() != last_dim) {
        throw std::invalid_argument("RMSNorm gamma size mismatch");
    }

    size_t outer = t.size() / last_dim;
    Tensor result(shape);
    for (size_t i = 0; i < outer; ++i) {
        float* slice = result.data() + i * last_dim;
        std::copy(t.data() + i * last_dim, t.data() + (i + 1) * last_dim, slice);
        rms_norm_inplace(slice, gamma.data(), last_dim, eps);
    }
    return result;
}


static float sigmoid(float x) {

    //we put the value between 0 and 1

    return 1.0f / (1.0f + std::exp(-x));
}

Tensor silu(const Tensor& t) {

    //multiply the original value with their own sigmoid value

    Tensor result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) {
        float x = t[i];
        result[i] = x * sigmoid(x);
    }
    return result;
}

} // namespace ops
