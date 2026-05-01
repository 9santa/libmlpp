#pragma once
#include "../core/tensor.h"
#include "../core/module.h"
#include "../core/parameter.h"
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nn {

class Conv2D : public Module {
private:
    Parameter W_;       // [KH, KW]
    Parameter b_;       // [1]

    size_t kernelH_;
    size_t kernelW_;
    size_t stride_;
    size_t padding_;

    Tensor inputCache_; // [H, W]

    // Padding is implicit via bounds checking
    // Indices outside are treated as zeros
    static bool inside(size_t i, size_t j, size_t H, size_t W) {
        return i >= 0 && j >= 0 && i < H && j < W;
    }

public:
    explicit Conv2D(size_t kernel_size, size_t stride = 1, size_t padding = 0, unsigned int seed = 42)
        : W_(std::vector<size_t>{kernel_size, kernel_size}),
          b_(std::vector<size_t>{1}),
          kernelH_(kernel_size),
          kernelW_(kernel_size),
          stride_(stride),
          padding_(padding) {

        if (kernel_size == 0) {
            throw std::runtime_error("Conv2D kernel size must be positive");
        }
        if (stride_ == 0) {
            throw std::runtime_error("Conv2D stride must be positive");
        }

        // 'He' random weight initialiation
        const double nin = static_cast<double>(kernel_size * kernel_size);
        const double stddev = std::sqrt(2.0 / nin);

        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, stddev);

        for (size_t i = 0; i < W_.value.numel(); i++) {
            W_.value[i] = dist(rng);
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 2) {
            throw std::runtime_error("Conv2D expects 2D input [H, W]");
        }

        const size_t H = x.shape()[0], W = x.shape()[1];
        const size_t paddedH = H + 2 * padding_, paddedW = W + 2 * padding_;

        if (paddedH < kernelH_ || paddedW < kernelW_) {
            throw std::runtime_error("Conv2D kernel larger than padded input");
        }

        const size_t outH = (paddedH - kernelH_) / stride_ + 1;
        const size_t outW = (paddedW - kernelW_) / stride_ + 1;

        inputCache_ = x;

        Tensor y({outH, outW}, 0.0);

        for (size_t oh = 0; oh < outH; oh++) {
            for (size_t ow = 0; ow < outW; ow++) {
                double sum = b_.value[0];

                for (size_t kh = 0; kh < kernelH_; kh++) {
                    for (size_t kw = 0; kw < kernelW_; kw++) {
                        size_t ih = (oh * stride_ + kh) - padding_;
                        size_t iw = (ow * stride_ + kw) - padding_;

                        if (inside(ih, iw, H, W)) {
                            sum += x.at(ih, iw) * W_.value.at(kh, kw);
                        }
                    }
                }

                y.at(oh, ow) = sum;
            }
        }

        return y;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 2) {
            throw std::runtime_error("Conv2D backward called before forward");
        }

        const auto& X_shape = inputCache_.shape();
        const auto& K_shape = W_.value.shape();
        const auto& G_shape = gradOutput.shape();

        const size_t H = X_shape[0], W = X_shape[1];
        const size_t KH = K_shape[0], KW = K_shape[1];
        const size_t OH = H - KH + 1, OW = W - KW + 1;

        if (G_shape.size() != 2 || G_shape[0] != OH || G_shape[1] != OW) {
            throw std::runtime_error("Conv2D gradOutput shape mismatch");
        }

        Tensor gradInput({H, W}, 0.0);

        for (size_t i = 0; i < OH; i++) {
            for (size_t j = 0; j < OW; j++) {
                const double go = gradOutput.at(i, j);

                // dL/db += dL/dY[i,j]
                b_.grad[0] += go;

                for (size_t kh = 0; kh < KH; kh++) {
                    for (size_t kw = 0; kw < KW; kw++) {
                        // Y[i,j] += X[i+kh, j+kw] * W[kh,kw]

                        int ih = (i * stride_ + kh) - padding_;
                        int iw = (j * stride_ + kw) - padding_;

                        if (inside(ih, iw, H, W)) {
                            // dL/dW[kh,kw] += dL/dY[i, j] * X[i+kh, j+kw]
                            W_.grad.at(kh, kw) += go * inputCache_.at(ih, iw);

                            // dL/dX[i+kh, j+kw] += dL/dY[i, j] * W[kh,kw]
                            gradInput.at(ih, iw) += go * W_.value.at(kh, kw);
                        }
                    }
                }
            }
        }

        return gradInput;
    }

    // 2D cross-correlation
    // X: [H, W]
    // K: [KH, KW]
    // Y: [H - KH + 1, W - KW + 1]
    Tensor corr2d(const Tensor& X, const Tensor& K) {
        const auto& X_shape = X.shape();
        const auto& K_shape = K.shape();

        if (X_shape.size() != 2 || K_shape.size() != 2) {
            throw std::runtime_error("corr2d expects 2D tensors");
        }

        size_t h = K_shape[0], w = K_shape[1];
        Tensor Y({X_shape[0] - h + 1, X_shape[1] - w + 1}, 0.0);
        for (size_t i = 0; i < Y.shape()[0]; i++) {
            for (size_t j = 0; j < Y.shape()[1]; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < h; k++) {
                    for (size_t l = 0; l < w; l++) {
                        sum += X.at(i + k, j + l) * K.at(k, l);
                    }
                }
                Y.at(i, j) = sum + b_.value[0];
            }
        }

        return Y;
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
