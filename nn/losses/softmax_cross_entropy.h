#pragma once
#include "../core/tensor.h"
#include <cmath>
#include <pthread.h>
#include <stdexcept>

namespace nn {

class SoftmaxCrossEntropy {
private:
    Tensor probsCache_;  // [batch, classes]
    Tensor targetCache_; // [batch] with class indices
    double eps_ = 1e-12;

public:
    double forward(const Tensor& logits, const Tensor& target) {
        if (logits.ndim() != 2) {
            throw std::runtime_error("SoftmaxCrossEntropy expects 2D logits [batch, classes]");
        }
        if (target.ndim() != 1) {
            throw std::runtime_error("SoftmaxCrossEntropy expects 1D target [batch]");
        }
        if (logits.shape()[0] != target.shape()[0]) {
            throw std::runtime_error("SoftmaxCrossEntropy batch size mismatch");
        }

        const size_t batch_size = logits.shape()[0];
        const size_t classes = logits.shape()[1];

        probsCache_ = Tensor({batch_size, classes}, 0.0);
        targetCache_ = target;

        double loss = 0.0;

        for (size_t n = 0; n < batch_size; n++) {
            double maxLogit = logits.at(n, 0);
            for (size_t c = 1; c < classes; c++) {
                maxLogit = std::max(maxLogit, logits.at(n, c));
            }

            double sumExp = 0.0;
            for (size_t c = 0; c < classes; c++) {
                const double ex = std::exp(logits.at(n, c) - maxLogit); // -maxLogit is a numerica stability trick
                probsCache_.at(n, c) = ex;
                sumExp += ex;
            }

            for (size_t c = 0; c < classes; c++) {
                probsCache_.at(n, c) /= sumExp;
            }

            const size_t y = static_cast<size_t>(target[n]);
            if (y >= classes) {
                throw std::runtime_error("SoftmaxCrossEntropy target class out of range");
            }

            loss += -std::log(std::max(probsCache_.at(n, y), eps_));
        }

        return loss / static_cast<double>(batch_size);
    }

    Tensor backward() const {
        Tensor grad = probsCache_;
        const size_t batch = probsCache_.shape()[0];
        const size_t classes = probsCache_.shape()[1];

        for (size_t n = 0; n < batch; n++) {
            // gradient of SoftmaxCrossEntropy with respect to logits: prob - y_onehot
            const size_t y = static_cast<size_t>(targetCache_[n]);
            grad.at(n, y) -= 1.0;
        }

        const double scale = 1.0 / static_cast<double>(batch);
        for (size_t i = 0; i < grad.numel(); i++) {
            grad[i] *= scale;
        }

        return grad;
    }
};

} // namespace nn
