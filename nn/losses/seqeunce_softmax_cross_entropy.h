#pragma once

#include "nn/core/tensor.h"

#include <cmath>
#include <stdexcept>


namespace nn {

class SequenceSoftmaxCrossEntropy {
private:
    Tensor probsCache_;  // [B, T, vocab_size]
    Tensor targetCache_; // [B, T]

    double ignoreLabel_;
    double eps_ = 1e-12;
    size_t validCount_ = 0;

public:
    explicit SequenceSoftmaxCrossEntropy(double ignoreLabel = -1.0)
        : ignoreLabel_(ignoreLabel) {}

    double forward(const Tensor& logits, const Tensor& target) {
        if (logits.ndim() != 3) {
            throw std::runtime_error("SequenceSoftmaxCrossEntropy expects logits [B, T, V]");
        }
        if (target.ndim() != 2) {
            throw std::runtime_error("SequenceSoftmaxCrossEntropy expects target [B, T]");
        }

        const size_t B = logits.shape()[0];
        const size_t T = logits.shape()[1];
        const size_t V = logits.shape()[2];

        if (target.shape()[0] != B || target.shape()[1] != T) {
            throw std::runtime_error("SequenceSoftmaxCrossEntropy target shape mismatch");
        }

        probsCache_ = Tensor({B, T, V}, 0.0);
        targetCache_ = target;
        validCount_ = 0;

        double loss = 0.0;

        for (size_t n = 0; n < B; n++) {
            for (size_t t = 0; t < T; t++) {
                const double rawLabel = target.at(n, t);

                double maxLogit = logits.at(n, t, 0);
                for (size_t v = 1; v < V; v++) {
                    maxLogit = std::max(maxLogit, logits.at(n, t, v));
                }

                double sumExp = 0.0;
                for (size_t v = 0; v < V; v++) {
                    double ex = std::exp(logits.at(n, t, v) - maxLogit);
                    probsCache_.at(n, t, v) = ex;
                    sumExp += ex;
                }

                for (size_t v = 0; v < V; v++) {
                    probsCache_.at(n, t, v) /= sumExp;
                }

                if (rawLabel == ignoreLabel_) continue;

                if (rawLabel < 0.0 || std::floor(rawLabel) != rawLabel) {
                    throw std::runtime_error("Sequence target must contain integer class ids");
                }

                size_t y = static_cast<double>(rawLabel);
                if (y >= V) {
                    throw std::runtime_error("Sequence target class out of range");
                }

                loss += -std::log(std::max(probsCache_.at(n, t, y), eps_));
                validCount_++;
            }
        }

        if (validCount_ == 0) {
            throw std::runtime_error("SequenceSoftmaxCrossEntropy has no valid targets");
        }

        return loss / static_cast<double>(validCount_);
    }

    Tensor backward() const {
        if (validCount_ == 0) {
            throw std::runtime_error("SequenceSoftmaxCrossEntropy backward called before valid forward");
        }

        Tensor grad = probsCache_;

        const size_t B = probsCache_.shape()[0];
        const size_t T = probsCache_.shape()[1];
        const size_t V = probsCache_.shape()[2];

        for (size_t n = 0; n < B; n++) {
            for (size_t t = 0; t < T; t++) {
                const double rawLabel = targetCache_.at(n, t);

                if (rawLabel == ignoreLabel_) {
                    for (size_t v = 0; v < V; v++) {
                        grad.at(n, t, v) = 0.0;
                    }
                    continue;
                }

                size_t y = static_cast<size_t>(rawLabel);
                grad.at(n, t, y) -= 1.0;
            }
        }

        const double scale = 1.0 / static_cast<double>(validCount_);
        for (size_t i = 0; i < grad.numel(); i++) {
            grad[i] *= scale;
        }

        return grad;
    }
};


} // namespace nn
