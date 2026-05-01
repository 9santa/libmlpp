#pragma once
#include "../core/module.h"
#include <stdexcept>

namespace nn {

class ReLU : public Module {
private:
    Tensor inputCache_;

public:
    Tensor forward(const Tensor& x) override {
        inputCache_ = x;
        Tensor out = x;
        for (size_t i = 0; i < out.numel(); i++) {
            out[i] = (out[i] > 0.0) ? out[i] : 0.0;
        }
        return out;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.numel() != inputCache_.numel()) {
            throw std::runtime_error("ReLU backward shape mismatch");
        }

        Tensor grad = gradOutput;
        for (size_t i = 0; i < grad.numel(); i++) {
            grad[i] = (inputCache_[i] > 0.0) ? grad[i] : 0.0;
        }
        return grad;
    }
};

} // namespace nn
