#include <iostream>
#include <cassert>
#include <cstring>
#include "arenaAllocator.hpp"
#include "tensor.hpp"

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

static void alloc_returns_nonnull() {
    Arena arena(1024);
    void* p = arena.alloc(64);
    if (!p) throw std::runtime_error("alloc returned null");
}

static void alloc_returns_aligned_memory() {
    Arena arena(1024);
    void* p = arena.alloc(64);
    if (reinterpret_cast<uintptr_t>(p) % alignof(float) != 0) {
        throw std::runtime_error("pointer not aligned to float");
    }
}

static void alloc_writes_and_reads_correctly() {
    Arena arena(1024);
    float* data = static_cast<float*>(arena.alloc(5 * sizeof(float)));
    for (int i = 0; i < 5; ++i) data[i] = static_cast<float>(i);
    for (int i = 0; i < 5; ++i) {
        if (data[i] != static_cast<float>(i)) {
            throw std::runtime_error("data mismatch after write");
        }
    }
}

static void multiple_allocations_dont_overlap() {
    Arena arena(1024);
    float* a = static_cast<float*>(arena.alloc(4 * sizeof(float)));
    float* b = static_cast<float*>(arena.alloc(4 * sizeof(float)));

    for (int i = 0; i < 4; ++i) a[i] = 1.0f;
    for (int i = 0; i < 4; ++i) b[i] = 2.0f;

    for (int i = 0; i < 4; ++i) {
        if (a[i] != 1.0f) throw std::runtime_error("first block corrupted");
        if (b[i] != 2.0f) throw std::runtime_error("second block corrupted");
    }
}

static void zero_size_alloc_returns_valid_ptr() {
    Arena arena(1024);
    void* p = arena.alloc(0);
    if (!p) throw std::runtime_error("zero-size alloc returned null");
}

static void reset_reuses_memory() {
    Arena arena(1024);
    void* p1 = arena.alloc(128);
    arena.reset();
    void* p2 = arena.alloc(128);
    if (p1 != p2) {
        throw std::runtime_error("reset did not reuse memory");
    }
}

static void alloc_exact_size() {
    Arena arena(256);
    arena.alloc(256);
    // should not throw
}

/*

before, alloc_too_large_throws does not pass test bcs 
the first version did not verify buffer_ limits 
*/
static void alloc_too_large_throws() {
    Arena arena(256);
    try {
        arena.alloc(512);
        throw std::runtime_error("expected out_of_range but no exception");
    } catch (const std::out_of_range&) {
        // expected
    }
}

static void sequential_alloc_increases_offset() {
    Arena arena(1024);
    void* p1 = arena.alloc(64);
    void* p2 = arena.alloc(64);
    if (p2 <= p1) {
        throw std::runtime_error("sequential allocations not monotonic");
    }
}

static void multiple_resets_work() {
    Arena arena(512);
    for (int r = 0; r < 10; ++r) {
        for (int i = 0; i < 5; ++i) {
            float* p = static_cast<float*>(arena.alloc(8 * sizeof(float)));
            p[0] = static_cast<float>(r * 100 + i);
        }
        arena.reset();
    }
    // if we reach here, resets succeeded
}

static void tensor_uses_arena() {
    Arena arena(4096);
    Tensor t({4, 64}, arena);
    if (t.size() != 256) throw std::runtime_error("tensor size mismatch");
    if (t.shape()[0] != 4) throw std::runtime_error("tensor dim 0 mismatch");
    if (t.shape()[1] != 64) throw std::runtime_error("tensor dim 1 mismatch");
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = 1.0f;
    }
    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i] != 1.0f) throw std::runtime_error("tensor arena data mismatch");
    }
}

static void tensor_arena_and_owned_coexist() {
    Arena arena(4096);
    Tensor owned({2, 8});
    Tensor from_arena({2, 8}, arena);
    for (size_t i = 0; i < owned.size(); ++i) owned[i] = 1.0f;
    for (size_t i = 0; i < from_arena.size(); ++i) from_arena[i] = 2.0f;
    for (size_t i = 0; i < owned.size(); ++i) {
        if (owned[i] != 1.0f) throw std::runtime_error("owned tensor corrupted");
        if (from_arena[i] != 2.0f) throw std::runtime_error("arena tensor corrupted");
    }
}

int main() {
    std::cout << "=== Arena Allocator Tests ===" << std::endl;

    TEST(alloc_returns_nonnull);
    TEST(alloc_returns_aligned_memory);
    TEST(alloc_writes_and_reads_correctly);
    TEST(multiple_allocations_dont_overlap);
    TEST(zero_size_alloc_returns_valid_ptr);
    TEST(reset_reuses_memory);
    TEST(alloc_exact_size);
    TEST(alloc_too_large_throws);
    TEST(sequential_alloc_increases_offset);
    TEST(multiple_resets_work);
    TEST(tensor_uses_arena);
    TEST(tensor_arena_and_owned_coexist);

    std::cout << std::endl;
    std::cout << "Results: " << (tests_passed + tests_failed)
              << " tests, " << tests_passed << " passed, "
              << tests_failed << " failed." << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
