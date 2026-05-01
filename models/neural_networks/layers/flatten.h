#pragma once

#include "../core/module.h"
#include <stdexcept>

namespace nn {

class Flatten : public Module {
private:
    std::vector<size_t> inputShape_;

public:
    Tensor forward(const Tensor& x) override {
        if (x.ndim() < 2) {
            throw std::runtime_error("Flatten expects input with batch dimension");
        }

        inputShape_ = x.shape();

        size_t features = 1;
        for (size_t i = 1; i < inputShape_.size(); i++) {
            features *= inputShape_[i];
        }

        Tensor out = x;
        out.reshapeInPlace({inputShape_[0], features});
        return out;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor grad = gradOutput;
        grad.reshapeInPlace(inputShape_);
        return grad;
    }

};

} // namespace nn
