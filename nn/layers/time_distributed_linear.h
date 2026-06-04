#pragma once

#include "nn/core/module.h"
#include "nn/core/tensor.h"
#include "nn/core/tensor_ops.h"
#include "nn/layers/linear.h"
#include <stdexcept>


namespace nn {

class TimeDistributedLinear : public Module {
private:
    Linear linear_;

    size_t cachedB_ = 0;
    size_t cachedT_ = 0;
    size_t cachedH_ = 0;

public:
    TimeDistributedLinear(size_t inFeatures,
                          size_t outFeatures,
                          unsigned int seed = 42)
        : linear_(inFeatures, outFeatures, seed) {}

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 3) {
            throw std::runtime_error("TimeDistributedLinear expects [B, T, H]");
        }

        cachedB_ = x.shape()[0];
        cachedT_ = x.shape()[1];
        cachedH_ = x.shape()[2];

        Tensor flat = flattenBatchTime(x);         // [B * T, H]
        Tensor flatOut = linear_.forward(flat); // [B * T, vocab_size]

        return unflattenBatchTime(flatOut, cachedB_, cachedT_);
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.ndim() != 3) {
            throw std::runtime_error("TimeDistributedLinear backward expects [B, T, vocab_size]");
        }

        if (gradOutput.shape()[0] != cachedB_ ||
            gradOutput.shape()[1] != cachedT_) {
            throw std::runtime_error("TimeDistributedLinear gradOutput shape mismatch");
        }

        Tensor flatGradOutput = flattenBatchTime(gradOutput); // [B * T, vocab_size]
        Tensor flatGradInput = linear_.backward(flatGradOutput); // [B * T, H]

        return unflattenBatchTime(flatGradInput, cachedB_, cachedT_);
    }

    std::vector<Parameter*> parameters() override {
        return linear_.parameters();
    }

    void zeroGrad() override {
        linear_.zeroGrad();
    }

    void train() override {
        training_ = true;
        linear_.train();
    }

    void eval() override {
        training_ = false;
        linear_.eval();
    }

    Linear& linear() {
        return linear_;
    }

    const Linear& linear() const {
        return linear_;
    }
};


} // namespace nn
