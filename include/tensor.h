#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include "arenaAllocator.h"

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<size_t> shape)
        : data_(new float[total_elements(shape)]())
        , size_(total_elements(shape))
        , shape_(std::move(shape))
        , strides_(compute_strides(shape_))
        , owns_data_(true)
    {}

    Tensor(std::vector<size_t> shape, std::vector<float> data)
        : data_(new float[data.size()])
        , size_(data.size())
        , shape_(std::move(shape))
        , strides_(compute_strides(shape_))
        , owns_data_(true)
    {
        size_t expected = total_elements(shape_);
        if (size_ != expected) {
            delete[] data_;
            data_ = nullptr;
            throw std::invalid_argument("data size does not match shape");
        }
        std::copy(data.begin(), data.end(), data_);
    }

    Tensor(std::vector<size_t> shape, Arena& arena)
        : data_(static_cast<float*>(arena.alloc(total_elements(shape) * sizeof(float))))
        //requesting storage to arenaAllocator
        , size_(total_elements(shape))
        , shape_(std::move(shape))
        , strides_(compute_strides(shape_))
        , owns_data_(false)
    {}

    ~Tensor() {
        if (owns_data_) delete[] data_;
    }

    Tensor(Tensor&& other) noexcept
        : data_(std::exchange(other.data_, nullptr))
        , size_(std::exchange(other.size_, 0))
        , shape_(std::move(other.shape_))
        , strides_(std::move(other.strides_))
        , owns_data_(std::exchange(other.owns_data_, false))
    {}

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owns_data_) delete[] data_;
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            owns_data_ = std::exchange(other.owns_data_, false);
        }
        return *this;
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    float& at(const std::vector<size_t>& indices) {
        return data_[offset(indices)];
    }

    const float& at(const std::vector<size_t>& indices) const {
        return data_[offset(indices)];
    }

    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }

    float* data() { return data_; }
    const float* data() const { return data_; }

    size_t size() const { return size_; }
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }

    size_t ndim() const { return shape_.size(); }
    size_t dim(size_t i) const { return shape_[i]; }

    Tensor view(std::vector<size_t> new_shape) const {
        size_t expected = total_elements(new_shape);
        if (expected != size_) {
            throw std::invalid_argument("new shape does not match element count");
        }
        Tensor t;
        t.data_ = data_;
        t.size_ = size_;
        t.shape_ = std::move(new_shape);
        t.strides_ = compute_strides(t.shape_);
        t.owns_data_ = false;
        return t;
    }

    bool empty() const { return size_ == 0; }

private:
    float* data_ = nullptr;
    size_t size_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    bool owns_data_ = false;

    size_t offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::out_of_range("index dimension mismatch");
        }
        size_t off = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            off += indices[i] * strides_[i];
        }
        return off;
    }

    static size_t total_elements(const std::vector<size_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
    }

    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides.back() = 1;
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
        return strides;
    }
};
