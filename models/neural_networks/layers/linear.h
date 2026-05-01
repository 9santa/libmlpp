#pragma once
#include "../core/module.h"
#include "models/neural_networks/core/tensor.h"
#include <random>
#include <stdexcept>

namespace nn {

class Linear : public Module {
private:
    size_t inFeatures_;
    size_t outFeatures_;

    Parameter W_; // [out, in]
    Parameter b_; // [out]

    Tensor inputCache_; // [batch, in]

public:
    Linear(size_t inFeatures, size_t outFeatures, unsigned int seed = 42)
        : inFeatures_(inFeatures), outFeatures_(outFeatures),
          W_(std::vector<size_t>{outFeatures, inFeatures}),
          b_(std::vector<size_t>{outFeatures}) {
        // Xavier weight initialization
        std::mt19937 rng(seed);
        const double limit = std::sqrt(6.0 / static_cast<double>(inFeatures_ + outFeatures_));
        std::uniform_real_distribution<double> dist(-limit, +limit);

        for (size_t o = 0; o < outFeatures_; o++) {
            for (size_t i = 0; i < inFeatures_; i++) {
                W_.value.at(o, i) = dist(rng);
            }
            b_.value[o] = 0.0;
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 2) {
            throw std::runtime_error("Linear expects 2D input [batch, features]");
        }
        if (x.shape()[1] != inFeatures_) {
            throw std::runtime_error("Linear input feature dimension mismatch");
        }

        inputCache_ = x;

        const size_t batch_size = x.shape()[0];
        Tensor out({batch_size, outFeatures_}, 0.0);

        for (size_t n = 0; n < batch_size; n++) {
            for (size_t o = 0; o < outFeatures_; o++) {
                double sum = b_.value[o];
                for (size_t i = 0; i < inFeatures_; i++) {
                    sum += x.at(n, i) * W_.value.at(o, i);
                }
                out.at(n, o) = sum;
            }
        }

        return out;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.ndim() != 2) {
            throw std::runtime_error("Linear backward expects 2D gradOutput");
        }
        if (gradOutput.shape()[1] != outFeatures_) {
            throw std::runtime_error("Linear gradOutput feature dimension mismatch");
        }

        const size_t batch_size = gradOutput.shape()[0];
        Tensor gradInput({batch_size, inFeatures_}, 0.0);

        for (size_t n = 0; n < batch_size; n++) {
            for (size_t o = 0; o < outFeatures_; o++) {
                const double gO = gradOutput.at(n, o);

                b_.grad[o] += gO;

                // chain rule
                for (size_t i = 0; i < inFeatures_; i++) {
                    W_.grad.at(o, i) += gO * inputCache_.at(n, i);
                    gradInput.at(n, i) += gO * W_.value.at(o, i);
                }
            }
        }

        return gradInput;
    }

    std::vector<Parameter*> parameters() override {
        return {&W_, &b_};
    }

    void zeroGrad() override {
        W_.zeroGrad();
        b_.zeroGrad();
    }

    Parameter& weight() {
        return W_;
    }

    Parameter& bias() {
        return b_;
    }

    const Parameter& weight() const {
        return W_;
    }

    const Parameter& bias() const {
        return b_;
    }

};

} // namespace nn
