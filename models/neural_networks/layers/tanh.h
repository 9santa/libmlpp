#pragma once

#include "../core/module.h"
#include <cmath>
#include <stdexcept>

namespace nn {

class Tanh : public Module {
private:
    Tensor outputCache_;

public:
    Tensor forward(const Tensor& x) override {
        outputCache_ = x;
        for (size_t i = 0; i < outputCache_.numel(); i++) {
            outputCache_[i] = std::tanh(outputCache_[i]);
        }
        return outputCache_;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.numel() != outputCache_.numel()) {
            throw std::runtime_error("Tanh backward shape mismatch");
        }

        Tensor grad = gradOutput;
        for (size_t i = 0; i < grad.numel(); i++) {
            const double y = outputCache_[i];
            grad[i] *= (1.0 - y * y);
        }
        return grad;
    }
};

} // namespace nn
