#pragma once

#include "../core/tensor.h"
#include "../core/module.h"
#include "../core/parameter.h"

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nn {

class Conv4D : public Module {
private:
    size_t inChannels_;
    size_t outChannels_;
    size_t kernelH_;
    size_t kernelW_;
    size_t stride_;
    size_t padding_;

    Parameter W_; // [outChannels, inChannels, KH, KW]
    Parameter b_; // [outChannels]

    Tensor inputCache_; // [N, C_in, H, W]

    // Padding is implicit via bounds checking
    // Indices outside are treated as zeros
    static bool inside(size_t i, size_t j, size_t H, size_t W) {
        return i >= 0 && j >= 0 && i < H && j < W;
    }

    static size_t outputSize(size_t inputSize,
                             size_t kernelSize,
                             size_t stride,
                             size_t padding) {
        const size_t padded = inputSize + 2 * padding;

        if (padded < kernelSize) {
            throw std::runtime_error("Conv4D kernel larger than padded input");
        }

        return (padded - kernelSize) / stride + 1;
    }

public:
    explicit Conv4D(size_t inChannels,
           size_t outChannels,
           size_t kernelSize,
           size_t stride = 1,
           size_t padding = 0,
           unsigned int seed = 42)
        : inChannels_(inChannels),
          outChannels_(outChannels),
          kernelH_(kernelSize),
          kernelW_(kernelSize),
          stride_(stride),
          padding_(padding),
          W_(std::vector<size_t>{outChannels, inChannels, kernelSize, kernelSize}),
          b_(std::vector<size_t>{outChannels}) {

        if (inChannels_ == 0 || outChannels_ == 0) {
            throw std::runtime_error("Conv4D channels must be positive");
        }
        if (kernelSize == 0) {
            throw std::runtime_error("Conv4D kernel size must be positive");
        }
        if (stride_ == 0) {
            throw std::runtime_error("Conv4D stride must be positive");
        }

        // 'He' initialization
        const double nin = static_cast<double>(inChannels_ * kernelH_ * kernelW_);
        const double stddev = std::sqrt(2.0 / nin);

        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, stddev);

        for (size_t i = 0; i < W_.value.numel(); i++) {
            W_.value[i] = dist(rng);
        }

        b_.value.fill(0.0);
    }


    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 4) {
            throw std::runtime_error("Conv4D expects input [N, C, H, W]");
        }

        const auto& xs = x.shape();

        const size_t N = xs[0];
        const size_t C = xs[1];
        const size_t H = xs[2];
        const size_t W = xs[3];

        if (C != inChannels_) {
            throw std::runtime_error("Conv4D input channel mismatch");
        }

        const size_t OH = outputSize(H, kernelH_, stride_, padding_);
        const size_t OW = outputSize(W, kernelW_, stride_, padding_);

        Tensor y({N, outChannels_, OH, OW}, 0.0);

        inputCache_ = x;

        for (size_t n = 0; n < N; ++n) {
            for (size_t oc = 0; oc < outChannels_; ++oc) {
                for (size_t oh = 0; oh < OH; ++oh) {
                    for (size_t ow = 0; ow < OW; ++ow) {
                        double sum = b_.value[oc];

                        for (size_t ic = 0; ic < inChannels_; ++ic) {
                            for (size_t kh = 0; kh < kernelH_; ++kh) {
                                for (size_t kw = 0; kw < kernelW_; ++kw) {
                                    const int ih = (oh * stride_ + kh) - padding_;
                                    const int iw = (ow * stride_ + kw) - padding_;

                                    if (inside(ih, iw, H, W)) {
                                        sum += x.at(n, ic, ih,iw)
                                             * W_.value.at(oc, ic, kh, kw);
                                    }
                                }
                            }
                        }

                        y.at(n, oc, oh, ow) = sum;
                    }
                }
            }
        }

        return y;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 4) {
            throw std::runtime_error("Conv4D backward called before forward");
        }

        const auto& xs = inputCache_.shape();

        const size_t N = xs[0];
        const size_t C = xs[1];
        const size_t H = xs[2];
        const size_t W = xs[3];

        const size_t OH = outputSize(H, kernelH_, stride_, padding_);
        const size_t OW = outputSize(W, kernelW_, stride_, padding_);

        if (gradOutput.ndim() != 4 ||
            gradOutput.shape()[0] != N ||
            gradOutput.shape()[1] != outChannels_ ||
            gradOutput.shape()[2] != OH ||
            gradOutput.shape()[3] != OW) {
            throw std::runtime_error("Conv4D gradOutput shape mismatch");
        }

        Tensor gradInput({N, C, H, W}, 0.0);

        for (size_t n = 0; n < N; ++n) {
            for (size_t oc = 0; oc < outChannels_; ++oc) {
                for (size_t oh = 0; oh < OH; ++oh) {
                    for (size_t ow = 0; ow < OW; ++ow) {
                        const double go = gradOutput.at(n, oc, oh, ow);

                        b_.grad[oc] += go;

                        for (size_t ic = 0; ic < inChannels_; ++ic) {
                            for (size_t kh = 0; kh < kernelH_; ++kh) {
                                for (size_t kw = 0; kw < kernelW_; ++kw) {
                                    const int ih = (oh * stride_ + kh) - padding_;
                                    const int iw = (ow * stride_ + kw) - padding_;

                                    if (inside(ih, iw, H, W)) {
                                        W_.grad.at(oc, ic, kh, kw) +=
                                            go * inputCache_.at(n, ic, ih, iw);

                                        gradInput.at(n, ic, ih, iw) +=
                                            go * W_.value.at(oc, ic, kh, kw);
                                    }
                                }
                            }
                        }
                    }
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
