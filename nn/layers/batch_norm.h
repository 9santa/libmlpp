#pragma once

#include "nn/core/module.h"
#include "nn/core/parameter.h"
#include "nn/core/tensor.h"
#include <cmath>
#include <stdexcept>


namespace nn {

/* Batch Normalization after Linear layer */
class BatchNorm1D : public Module {
private:
    size_t numFeatures_;
    double eps_;
    double momentum_;

    Parameter gamma_; // [F], scale
    Parameter beta_;  // [F], shift

    Tensor runningMean_; // [F]
    Tensor runningVar_;  // [F]

    Tensor xhatCache_;   // [N, F], normalized version of the input
    Tensor invStdCache_; // [F]

    bool cacheValid_ = false;

public:
    BatchNorm1D(size_t numFeatures,
                double eps = 1e-5,
                double momentum = 0.9)
        : numFeatures_(numFeatures),
          eps_(eps),
          momentum_(momentum),
          gamma_(std::vector<size_t>{numFeatures}),
          beta_(std::vector<size_t>{numFeatures}),
          runningMean_(std::vector<size_t>{numFeatures}, 0.0),
          runningVar_(std::vector<size_t>{numFeatures}, 1.0),
          invStdCache_(std::vector<size_t>{numFeatures}, 0.0) {
        if (numFeatures_ == 0) {
            throw std::runtime_error("BatchNorm1D numFeatures must be positive");
        }

        gamma_.value.fill(1.0);
        beta_.value.fill(0.0);
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 2) {
            throw std::runtime_error("BatchNorm1D expects input [N, F]");
        }

        const size_t N = x.shape()[0], F = x.shape()[1];

        if (F != numFeatures_) {
            throw std::runtime_error("BatchNorm1D feature dimension mismatch");
        }
        if (N == 0) {
            throw std::runtime_error("BatchNorm1D empty batch");
        }

        Tensor y({N, F}, 0.0);

        if (training_) {
            cacheValid_ = true;
            xhatCache_ = Tensor({N, F}, 0.0);

            Tensor batchMean({F}, 0.0);
            Tensor batchVar({F}, 0.0);

            for (size_t f = 0; f < F; f++) {
                double mean = 0.0;
                for (size_t n = 0; n < N; n++) mean += x.at(n, f);
                mean /= static_cast<double>(N);
                batchMean[f] = mean;

                double var = 0.0;
                for (size_t n = 0; n < N; n++) {
                    const double diff = x.at(n, f) - mean;
                    var += diff * diff;
                }
                var /= static_cast<double>(N);
                batchVar[f] = var;

                invStdCache_[f] = 1.0 / std::sqrt(var + eps_);

                runningMean_[f] = momentum_ * runningMean_[f] + (1.0 - momentum_) * mean;
                runningVar_[f] = momentum_ * runningVar_[f] + (1.0 - momentum_) * var;
            }

            for (size_t n = 0; n < N; n++) {
                for (size_t f = 0; f < F; f++) {
                    const double xhat = (x.at(n, f) - batchMean[f]) * invStdCache_[f];
                    xhatCache_.at(n, f) = xhat;
                    y.at(n, f) = gamma_.value[f] * xhat + beta_.value[f];
                }
            }
        } else {
            cacheValid_ = false;

            for (size_t n = 0; n < N; n++) {
                for (size_t f = 0; f < F; f++) {
                    const double invStd = 1.0 / std::sqrt(runningVar_[f] + eps_);
                    const double xhat = (x.at(n, f) - runningMean_[f]) * invStd;
                    y.at(n, f) = gamma_.value[f] * xhat + beta_.value[f];
                }
            }
        }

        return y;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (!cacheValid_) {
            throw std::runtime_error("BatchNorm1D backward requires training forward pass");
        }

        if (gradOutput.ndim() != 2) {
            throw std::runtime_error("BatchNorm1D backward expects [N, F]");
        }

        const size_t N = gradOutput.shape()[0], F = gradOutput.shape()[1];

        if (F != numFeatures_) {
            throw std::runtime_error("BatchNorm1D gradOutput feature mismatch");
        }
        if (xhatCache_.shape() != gradOutput.shape()) {
            throw std::runtime_error("BatchNorm1D cache shape mismatch");
        }

        Tensor gradInput({N, F}, 0.0);
        const double M = 

    }
};


/* Batch Normalization after Conv2D (4D) layer */
class BatchNorm2D : public Module {

};




} // namespace nn
