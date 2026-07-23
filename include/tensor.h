#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<size_t> shape)
        : data_(std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{}))
        , shape_(std::move(shape))
        , strides_(compute_strides(shape_))
    {}

    Tensor(std::vector<size_t> shape, std::vector<float> data)
        : data_(std::move(data))
        , shape_(std::move(shape))
        , strides_(compute_strides(shape_))
    {
        size_t expected = std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<>{});
        if (data_.size() != expected) { //avoid unexpected states
            throw std::invalid_argument("data size does not match shape");
        }
    }

    float& at(const std::vector<size_t>& indices) {
        return data_[offset(indices)];
    }

    const float& at(const std::vector<size_t>& indices) const {
        return data_[offset(indices)];
    }

    float& operator[](size_t i) {
        return data_[i];
    }

    const float& operator[](size_t i) const {
        return data_[i];
    }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    size_t size() const { return data_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }

    size_t ndim() const { return shape_.size(); }
    size_t dim(size_t i) const { return shape_[i]; }

    Tensor view(std::vector<size_t> new_shape) const {
        size_t expected = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<>{});
        if (expected != size()) {
            throw std::invalid_argument("new shape does not match element count");
        }
        Tensor t;
        t.data_ = data_;
        t.shape_ = std::move(new_shape);
        t.strides_ = compute_strides(t.shape_);
        return t;
    }

    bool empty() const { return data_.empty(); }

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

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
