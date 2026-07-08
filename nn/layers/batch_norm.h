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

        // Compute gradients per feature independently
        for (size_t f = 0; f < F; f++) {
            double d_gamma = 0.0;
            double d_beta = 0.0;
            double d_mu = 0.0;
            double d_sigma2 = 0.0;

            const double invStd = invStdCache_[f]; // 1 / sqrt(sigma^2 + eps)
            const double gamma_f = gamma_.value[f];

            // PASS 1: Accumulate sums for Eq 2, 3, 5 and 6
            for (size_t i = 0; i < N; i++) {
                const double go = gradOutput.at(i, f); // dL/dy_i
                const double xhat = xhatCache_.at(i, f); // xhat_i

                // Eq 2 & 3: wrt gamma and beta
                d_gamma += go * xhat;
                d_beta += go;

                // Eq 4: wrt xhat
                const double d_xhat = go * gamma_f;

                // accumulate sum for Eq 5 (wrt mu)
                d_mu += d_xhat;
                // accumulate sum for Eq 6 (wrt sigma^2)
                // (x_i - mu) = xhat_i / invStd
                const double x_minus_mu = xhat / invStd;
                d_sigma2 += d_xhat * x_minus_mu;
            }

            // finalize Eq 5: dL/d_mu = -1 / sqrt(sigma^2 + eps) * sum(dL/d_xhat_i)
            d_mu = d_mu * (-invStd);

            // finalize Eq 6: dL/d_sigma2 = -0.5 * (sigma^2 + eps)^(-3/2) * sum(dL/d_xhat_i * (x_i - mu))
            // invStd^3 is exactly (sigma^2 + eps)^(-3/2)
            d_sigma2 = d_sigma2 * (-0.5 * invStd * invStd * invStd);

            // apply BN's parameters gradients
            gamma_.grad[f] += d_gamma;
            beta_.grad[f] += d_beta;

            // PASS 2: Compute final gradient wrt x_i (Eq 7)
            for (size_t i = 0; i < N; i++) {
                const double go = gradOutput.at(i, f);
                const double xhat = xhatCache_.at(i, f);

                // Recompute Eq 4 locally
                const double d_xhat = go * gamma_f;
                const double x_minus_mu = xhat / invStd;

                // Eq 7 (wrt x) broken down into its three flowing gradient paths:
                const double term1 = d_xhat * invStd;
                const double term2 = d_sigma2 * (2.0 / N) * x_minus_mu;
                const double term3 = d_mu / N;

                gradInput.at(i, f) = term1 + term2 + term3;
            }
        }

        return gradInput;
    }
};


/* Batch Normalization after Conv2D (4D) layer */
class BatchNorm2D : public Module {
private:
    size_t numFeatures_; // Number of Channels (C)
    double eps_;
    double momentum_;

    Parameter gamma_; // [C], scale
    Parameter beta_;  // [C], shift

    Tensor runningMean_; // [C]
    Tensor runningVar_;  // [C]

    Tensor xhatCache_; // [N, C, H, W]
    Tensor invStdCache_; // [C]

    bool cacheValid_ = false;

public:
    BatchNorm2D(size_t numFeatures,
                double eps = 1e-5,
                double momentum = 0.9)
          : numFeatures_(numFeatures),
            eps_(eps),
            momentum_(momentum),
            gamma_(std::vector<size_t>{numFeatures}),
            beta_(std::vector<size_t>{numFeatures_}),
            runningMean_(std::vector<size_t>{numFeatures}, 0.0),
            runningVar_(std::vector<size_t>{numFeatures}, 1.0),
            invStdCache_(std::vector<size_t>{numFeatures}, 0.0) {
        if (numFeatures_ == 0) {
            throw std::runtime_error("BatchNorm2D numFeatures must be positive");
        }

        gamma_.value.fill(1.0);
        beta_.value.fill(0.0);
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 4) {
            throw std::runtime_error("BatchNorm2D expects input [N, C, H, W]");
        }

        const size_t N = x.shape()[0];
        const size_t C = x.shape()[1];
        const size_t H = x.shape()[2];
        const size_t W = x.shape()[3];

        if (C != numFeatures_) {
            throw std::runtime_error("BatchNorm2D channel dimension mismatch");
        }
        if (N == 0) {
            throw std::runtime_error("BatchNorm2D empty batch");
        }

        Tensor y({N, C, H, W}, 0.0);

        // The total number of elements normalized per channel
        const size_t n_elements = N * H * W;

        if (training_) {
            cacheValid_ = true;
            xhatCache_ = Tensor({N, C, H, W}, 0.0);

            for (size_t c = 0; c < C; c++) {
                // 1. Compute Mean
                double mean = 0.0;
                for (size_t n = 0; n < N; n++) {
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            mean += x.at(n, c, h, w);
                        }
                    }
                }
                mean /= (double)n_elements;

                // 2. Compute Variance
                double var = 0.0;
                for (size_t n = 0; n < N; n++) {
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            const double diff = x.at(n, c, h, w) - mean;
                            var += diff * diff;
                        }
                    }
                }
                var /= (double)n_elements;

                invStdCache_[c] = 1.0 / std::sqrt(var + eps_);

                // 3. Update running stats
                runningMean_[c] = momentum_ * runningMean_[c] + (1.0 - momentum_) * mean;
                runningVar_[c] = momentum_ * runningVar_[c] + (1.0 - momentum_) * var;

                // 4. Apply BatchNorm per channel
                for (size_t n = 0; n < N; n++) {
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            const double xhat = (x.at(n, c, h, w) - mean) * invStdCache_[c];
                            xhatCache_.at(n, c, h, w) = xhat;
                            y.at(n, c, h, w) = gamma_.value[c] * xhat + beta_.value[c];
                        }
                    }
                }
            }
        } else {
            // not training
            cacheValid_ = false;

            for (size_t c = 0; c < C; c++) {
                const double invStd = 1.0 / std::sqrt(runningVar_[c] + eps_);
                for (size_t n = 0; n < N; n++) {
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            const double xhat = (x.at(n, c, h, w) - runningMean_[c]) * invStd;
                            y.at(n, c, h, w) = gamma_.value[c] * xhat + beta_.value[c];
                        }
                    }
                }
            }
        }

        return y;
    }

    Tensor backward(const Tensor &gradOutput) override {
        if (!cacheValid_) {
            throw std::runtime_error("BatchNorm2D backward requires training forward pass");
        }

        if (gradOutput.ndim() != 4) {
            throw std::runtime_error("BatchNorm2D backward expects [N, C, H, W]");
        }

        const size_t N = gradOutput.shape()[0];
        const size_t C = gradOutput.shape()[1];
        const size_t H = gradOutput.shape()[2];
        const size_t W = gradOutput.shape()[3];

        if (C != numFeatures_) {
            throw std::runtime_error("BatchNorm2D gradOutput channel mismatch");
        }

        Tensor gradInput({N, C, H, W}, 0.0);
        const size_t n_elements = N * H * W;

        for (size_t c = 0; c < C; c++) {
            double d_gamma = 0.0;
            double d_beta = 0.0;
            double d_mu = 0.0;
            double d_sigma2 = 0.0;

            const double invStd = invStdCache_[c];
            const double gamma_c = gamma_.value[c];

            // PASS 1: Accumulate
            for (size_t n = 0; n < N; n++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        const double go = gradOutput.at(n, c, h, w);
                        const double xhat = xhatCache_.at(n, c, h, w);

                        d_gamma += go * xhat;
                        d_beta += go;

                        const double d_xhat = go * gamma_c;
                        d_mu += d_xhat;

                        const double x_minus_mu = xhat / invStd;
                        d_sigma2 += d_xhat * x_minus_mu;
                    }
                }
            }

            // Finalize d_mu and d_sigma2
            d_mu = d_mu * (-invStd);
            d_sigma2 = d_sigma2 * (-0.5 * invStd * invStd * invStd);

            gamma_.grad[c] += d_gamma;
            beta_.grad[c] += d_beta;

            // PASS 2: Compute final gradient wrt x
            for (size_t n = 0; n < N; n++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        const double go = gradOutput.at(n, c, h, w);
                        const double xhat = xhatCache_.at(n, c, h, w);

                        const double d_xhat = go * gamma_c;
                        const double x_minus_mu = xhat / invStd;

                        const double term1 = d_xhat * invStd;
                        const double term2 = d_sigma2 * (2.0 / n_elements) * x_minus_mu;
                        const double term3 = d_mu * (1.0 / n_elements);

                        gradInput.at(n, c, h, w) = term1 + term2 + term3;
                    }
                }
            }
        }

        return gradInput;
    }

    std::vector<Parameter*> parameters() override {
        return {&gamma_, &beta_};
    }

    void zeroGrad() override {
        gamma_.zeroGrad();
        beta_.zeroGrad();
    }

    Parameter& weight() { return gamma_; }
    Parameter& bias() { return beta_; }
    const Parameter& weight() const { return gamma_; }
    const Parameter& bias() const { return beta_; }
};


} // namespace nn
