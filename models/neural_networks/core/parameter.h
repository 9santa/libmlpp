#pragma once
#include "tensor.h"

namespace nn {

struct Parameter {
    Tensor value;
    Tensor grad;

    Parameter() = default;

    explicit Parameter(const std::vector<size_t>& shape)
            : value(shape, 0.0), grad(shape, 0.0) {}

    explicit Parameter(const Tensor& tensor)
            : value(tensor), grad(tensor.shape(), 0.0) {}

    void zeroGrad() {
        grad.fill(0.0);
    }
};

} // namespace nn
