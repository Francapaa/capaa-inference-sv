#include "ops.h"
#include "tensor.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { \
        try { \
            name(); \
            ++tests_passed; \
            std::cout << "  PASSED: " #name << std::endl; \
        } catch (const std::exception& e) { \
            ++tests_failed; \
            std::cout << "  FAILED: " #name " - " << e.what() << std::endl; \
        } catch (...) { \
            ++tests_failed; \
            std::cout << "  FAILED: " #name " - unknown error" << std::endl; \
        } \
    } while(0)

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// --- matmul ---

static void matmul_small() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor c = ops::matmul(a, b);
    if (c.shape() != std::vector<size_t>({2, 2}))
        throw std::runtime_error("shape mismatch");
    if (!approx(c.at({0, 0}), 58.0f)) throw std::runtime_error("c[0,0] wrong");
    if (!approx(c.at({0, 1}), 64.0f)) throw std::runtime_error("c[0,1] wrong");
    if (!approx(c.at({1, 0}), 139.0f)) throw std::runtime_error("c[1,0] wrong");
    if (!approx(c.at({1, 1}), 154.0f)) throw std::runtime_error("c[1,1] wrong");
}

static void matmul_1x1() {
    Tensor a({1, 1}, {3.0f});
    Tensor b({1, 1}, {4.0f});
    Tensor c = ops::matmul(a, b);
    if (!approx(c.at({0, 0}), 12.0f))
        throw std::runtime_error("1x1 matmul wrong");
}

static void matmul_dim_mismatch() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 2}, {7, 8, 9, 10});
    try {
        ops::matmul(a, b);
        throw std::runtime_error("expected invalid_argument");
    } catch (const std::invalid_argument&) {}
}

static void matmul_not_2d() {
    Tensor a({2, 3, 1});
    Tensor b({3, 2});
    try {
        ops::matmul(a, b);
        throw std::runtime_error("expected invalid_argument for 3D");
    } catch (const std::invalid_argument&) {}
}

// --- add ---

static void add_elementwise() {
    Tensor a({3}, {1.0f, 2.0f, 3.0f});
    Tensor b({3}, {4.0f, 5.0f, 6.0f});
    Tensor c = ops::add(a, b);
    if (!approx(c[0], 5.0f)) throw std::runtime_error("add[0] wrong");
    if (!approx(c[1], 7.0f)) throw std::runtime_error("add[1] wrong");
    if (!approx(c[2], 9.0f)) throw std::runtime_error("add[2] wrong");
}

static void add_shape_mismatch() {
    Tensor a({2, 3});
    Tensor b({3, 2});
    try {
        ops::add(a, b);
        throw std::runtime_error("expected invalid_argument");
    } catch (const std::invalid_argument&) {}
}

// --- mul ---

static void mul_scalar() {
    Tensor t({3}, {1.0f, 2.0f, 3.0f});
    Tensor r = ops::mul(t, 2.5f);
    if (!approx(r[0], 2.5f)) throw std::runtime_error("mul[0] wrong");
    if (!approx(r[1], 5.0f)) throw std::runtime_error("mul[1] wrong");
    if (!approx(r[2], 7.5f)) throw std::runtime_error("mul[2] wrong");
}

static void mul_negative() {
    Tensor t({2}, {1.0f, -2.0f});
    Tensor r = ops::mul(t, -3.0f);
    if (!approx(r[0], -3.0f)) throw std::runtime_error("neg mul[0] wrong");
    if (!approx(r[1], 6.0f))  throw std::runtime_error("neg mul[1] wrong");
}

static void mul_zero() {
    Tensor t({3}, {1.0f, 2.0f, 3.0f});
    Tensor r = ops::mul(t, 0.0f);
    for (size_t i = 0; i < 3; ++i)
        if (!approx(r[i], 0.0f))
            throw std::runtime_error("zero mul not zero");
}

// --- softmax ---

static void softmax_known() {
    Tensor t({3}, {1.0f, 2.0f, 3.0f});
    Tensor r = ops::softmax(t);
    if (!approx(r[0], 0.09003057f, 1e-4f))
        throw std::runtime_error("softmax[0] wrong");
    if (!approx(r[1], 0.24472847f, 1e-4f))
        throw std::runtime_error("softmax[1] wrong");
    if (!approx(r[2], 0.66524096f, 1e-4f))
        throw std::runtime_error("softmax[2] wrong");
}

static void softmax_single() {
    Tensor t({1}, {5.0f});
    Tensor r = ops::softmax(t);
    if (!approx(r[0], 1.0f))
        throw std::runtime_error("single softmax wrong");
}

static void softmax_zeros() {
    Tensor t({3}, {0.0f, 0.0f, 0.0f});
    Tensor r = ops::softmax(t);
    float expected = 1.0f / 3.0f;
    for (size_t i = 0; i < 3; ++i)
        if (!approx(r[i], expected, 1e-5f))
            throw std::runtime_error("zero softmax not uniform");
}

// --- RMSNorm ---

static void rmsnorm_known() {
    Tensor t({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor gamma({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor r = ops::RMSNorm(t, gamma, 1e-5f);

    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms = std::sqrt(sum_sq / 4.0f + 1e-5f);
    float inv_rms = 1.0f / rms;

    if (!approx(r[0], 1.0f * inv_rms, 1e-4f))
        throw std::runtime_error("rmsnorm[0] wrong");
    if (!approx(r[1], 2.0f * inv_rms, 1e-4f))
        throw std::runtime_error("rmsnorm[1] wrong");
    if (!approx(r[2], 3.0f * inv_rms, 1e-4f))
        throw std::runtime_error("rmsnorm[2] wrong");
    if (!approx(r[3], 4.0f * inv_rms, 1e-4f))
        throw std::runtime_error("rmsnorm[3] wrong");
}

static void rmsnorm_gamma_mismatch() {
    Tensor t({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor gamma({3}, {1.0f, 1.0f, 1.0f});
    try {
        ops::RMSNorm(t, gamma);
        throw std::runtime_error("expected invalid_argument");
    } catch (const std::invalid_argument&) {}
}

static void rmsnorm_empty_tensor() {
    Tensor t(std::vector<size_t>{});
    Tensor gamma({1});
    try {
        ops::RMSNorm(t, gamma);
        throw std::runtime_error("expected invalid_argument");
    } catch (const std::invalid_argument&) {}
}

// --- silu ---

static void silu_known() {
    Tensor t({3}, {0.0f, 1.0f, -2.0f});
    Tensor r = ops::silu(t);

    if (!approx(r[0], 0.0f, 1e-5f))
        throw std::runtime_error("silu(0) wrong");

    float sig1 = 1.0f / (1.0f + std::exp(-1.0f));
    if (!approx(r[1], 1.0f * sig1, 1e-4f))
        throw std::runtime_error("silu(1) wrong");

    float sig2 = 1.0f / (1.0f + std::exp(2.0f));
    if (!approx(r[2], -2.0f * sig2, 1e-4f))
        throw std::runtime_error("silu(-2) wrong");
}

static void silu_negative_input() {
    Tensor t({2}, {-5.0f, -0.5f});
    Tensor r = ops::silu(t);
    for (size_t i = 0; i < 2; ++i) {
        float expected = t[i] / (1.0f + std::exp(-t[i]));
        if (!approx(r[i], expected, 1e-4f))
            throw std::runtime_error("silu negative mismatch");
    }
}

// --- main ---

int main() {
    std::cout << "=== Ops Tests ===" << std::endl;

    TEST(matmul_small);
    TEST(matmul_1x1);
    TEST(matmul_dim_mismatch);
    TEST(matmul_not_2d);

    TEST(add_elementwise);
    TEST(add_shape_mismatch);

    TEST(mul_scalar);
    TEST(mul_negative);
    TEST(mul_zero);

    TEST(softmax_known);
    TEST(softmax_single);
    TEST(softmax_zeros);

    TEST(rmsnorm_known);
    TEST(rmsnorm_gamma_mismatch);
    TEST(rmsnorm_empty_tensor);

    TEST(silu_known);
    TEST(silu_negative_input);

    std::cout << std::endl;
    std::cout << "Results: " << (tests_passed + tests_failed)
              << " tests, " << tests_passed << " passed, "
              << tests_failed << " failed." << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
