#pragma once
#include "../core/module.h"
#include <cmath>

namespace nn {

class Sigmoid : public Module {
private:
    Tensor outputCache_;

    static double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

public:
    Tensor forward(const Tensor& x) override {
        outputCache_ = x;
        for (size_t i = 0; i < outputCache_.numel(); i++) {
            outputCache_[i] = sigmoid(outputCache_[i]);
        }
        return outputCache_;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.numel() != outputCache_.numel()) {
            throw std::runtime_error("Sigmoid backward shape mismatch");
        }

        Tensor grad = gradOutput;
        for (size_t i = 0; i < grad.numel(); i++) {
            const double y = outputCache_[i];
            grad[i] *= y * (1.0 - y);
        }
        return grad;
    }
};

} // namespace nn
