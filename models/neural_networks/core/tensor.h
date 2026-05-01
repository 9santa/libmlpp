#pragma once
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>


namespace nn {

/* Tensor object */
class Tensor {
private:
    std::vector<size_t> shape_;
    std::vector<double> data_;

    static size_t numelOf(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }

        return std::accumulate(shape.begin(), shape.end(),
                               static_cast<size_t>(1),
                               std::multiplies<size_t>()
                               );
    }

public:
    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape, double value = 0.0)
            : shape_(shape), data_(numelOf(shape), value) {}

    const std::vector<size_t>& shape() const { return shape_; }

    size_t ndim() const {
        return shape_.size();
    }

    size_t numel() const {
        return data_.size();
    }

    bool empty() const {
        return data_.empty();
    }

    const std::vector<double>& data() const { return data_; }

    void fill(double value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void reshapeInPlace(const std::vector<size_t>& newShape) {
        if (numelOf(newShape) != numel()) {
            throw std::runtime_error("Tensor reshape changes number of elemenets");
        }
        shape_ = newShape;
    }

    double& operator[](size_t i) {
        return data_.at(i);
    }

    const double& operator[](size_t i) const {
        return data_.at(i);
    }

    double& at(size_t i, size_t j) {
        if (shape_.size() != 2) {
            throw std::runtime_error("Tensor::at(i, j) requires 2D tensor");
        }
        return data_.at(i * shape_[1] + j);
    }

    const double& at(size_t i, size_t j) const {
        if (shape_.size() != 2) {
            throw std::runtime_error("Tensor::at(i, j) requires 2D tensor");
        }
        return data_.at(i * shape_[1] + j);
    }

    double& at(size_t i, size_t j, size_t k, size_t h) {
        if (shape_.size() != 4) {
            throw std::runtime_error("Tensor::at(i, j, k, h) requires 4D tensor");
        }

        const size_t J = shape_[1];
        const size_t K = shape_[2];
        const size_t H = shape_[3];

        return data_.at(((i * J + j) * K + k) * H + h);
    }

    const double& at(size_t i, size_t j, size_t k, size_t h) const {
        if (shape_.size() != 4) {
            throw std::runtime_error("Tensor::at(i, j, k, h) requires 4D tensor");
        }

        const size_t J = shape_[1];
        const size_t K = shape_[2];
        const size_t H = shape_[3];

        return data_.at(((i * J + j) * K + k) * H + h);
    }
};

} // namespace nn
