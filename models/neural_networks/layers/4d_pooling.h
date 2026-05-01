#pragma once

#include "../core/module.h"
#include "../core/tensor.h"
#include <stdexcept>

namespace nn {

enum class PoolMode {
    Max,
    Avg
};

class Pool4D : public Module {
private:
    size_t poolH_;
    size_t poolW_;
    size_t stride_;
    size_t padding_;
    PoolMode mode_;

    Tensor inputCache_; // [batch size, channels, H, W]

    // Used only for max pooling.
    // Stores flattened input index h * W + w for each output cell.
    std::vector<size_t> maxIndices_;

    static bool inside(size_t i, size_t j, size_t H, size_t W) {
        return i >= 0 && j >= 0 && i < H && j < W;
    }

    static size_t outputSize(size_t inputSize,
                             size_t poolSize,
                             size_t stride,
                             size_t padding) {
        const size_t padded = inputSize + 2 * padding;

        if (padded < poolSize) {
            throw std::runtime_error("Pool4D window larger than padded input");
        }

        return (padded - poolSize) / stride + 1;
    }

    static size_t flatOutputIndex(size_t n,
                                  size_t c,
                                  size_t oh,
                                  size_t ow,
                                  size_t C,
                                  size_t OH,
                                  size_t OW) {
        return ((n * C + c) * OH + oh) * OW + ow;
    }

public:
    explicit Pool4D(size_t poolSize,
                    size_t stride,
                    PoolMode mode = PoolMode::Max,
                    size_t padding = 0)
        : poolH_(poolSize),
          poolW_(poolSize),
          stride_(stride),
          padding_(padding),
          mode_(mode) {
        if (poolSize == 0) {
            throw std::runtime_error("Pool4D pool size must be positive");
        }
        if (stride_ == 0) {
            throw std::runtime_error("Pool4D stride must be positive");
        }
    }

    explicit Pool4D(size_t poolSize, PoolMode mode = PoolMode::Max) : Pool4D(poolSize, poolSize, mode, 0) {}

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 4) {
            throw std::runtime_error("Pool4D expects input [N, C, H, W]");
        }

        const auto& xs = x.shape();

        const size_t N = xs[0];
        const size_t C = xs[1];
        const size_t H = xs[2];
        const size_t W = xs[3];

        const size_t OH = outputSize(H, poolH_, stride_, padding_);
        const size_t OW = outputSize(W, poolW_, stride_, padding_);

        inputCache_ = x;

        Tensor y({N, C, OH, OW}, 0.0);

        if (mode_ == PoolMode::Max) {
            maxIndices_.assign(N * C * OH * OW, 0);
        } else {
            maxIndices_.clear();
        }

        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t oh = 0; oh < OH; ++oh) {
                    for (size_t ow = 0; ow < OW; ++ow) {
                        if (mode_ == PoolMode::Max) {
                            double best = std::numeric_limits<double>::lowest();
                            size_t bestIndex = 0;
                            bool found = false;

                            for (size_t kh = 0; kh < poolH_; ++kh) {
                                for (size_t kw = 0; kw < poolW_; ++kw) {
                                    const int ih =
                                        static_cast<int>(oh * stride_ + kh) -
                                        static_cast<int>(padding_);

                                    const int iw =
                                        static_cast<int>(ow * stride_ + kw) -
                                        static_cast<int>(padding_);

                                    if (inside(ih, iw, H, W)) {
                                        const size_t inputH = static_cast<size_t>(ih);
                                        const size_t inputW = static_cast<size_t>(iw);
                                        const double value = x.at(n, c, inputH, inputW);

                                        if (!found || value > best) {
                                            found = true;
                                            best = value;
                                            bestIndex = inputH * W + inputW;
                                        }
                                    }
                                }
                            }

                            if (!found) {
                                throw std::runtime_error("Pool4D max window contains no valid cells");
                            }

                            y.at(n, c, oh, ow) = best;

                            const size_t outIdx =
                                flatOutputIndex(n, c, oh, ow, C, OH, OW);

                            maxIndices_[outIdx] = bestIndex;
                        } else {
                            double sum = 0.0;
                            size_t count = 0;

                            for (size_t kh = 0; kh < poolH_; ++kh) {
                                for (size_t kw = 0; kw < poolW_; ++kw) {
                                    const int ih =
                                        static_cast<int>(oh * stride_ + kh) -
                                        static_cast<int>(padding_);

                                    const int iw =
                                        static_cast<int>(ow * stride_ + kw) -
                                        static_cast<int>(padding_);

                                    if (inside(ih, iw, H, W)) {
                                        sum += x.at(n, c,
                                                    static_cast<size_t>(ih),
                                                    static_cast<size_t>(iw));
                                        ++count;
                                    }
                                }
                            }

                            if (count == 0) {
                                throw std::runtime_error("Pool4D average window contains no valid cells");
                            }

                            y.at(n, c, oh, ow) = sum / static_cast<double>(count);
                        }
                    }
                }
            }
        }

        return y;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 4) {
            throw std::runtime_error("Pool4D backward called before forward");
        }

        const auto& xs = inputCache_.shape();

        const size_t N = xs[0];
        const size_t C = xs[1];
        const size_t H = xs[2];
        const size_t W = xs[3];

        const size_t OH = outputSize(H, poolH_, stride_, padding_);
        const size_t OW = outputSize(W, poolW_, stride_, padding_);

        if (gradOutput.ndim() != 4 ||
            gradOutput.shape()[0] != N ||
            gradOutput.shape()[1] != C ||
            gradOutput.shape()[2] != OH ||
            gradOutput.shape()[3] != OW) {
            throw std::runtime_error("Pool4D gradOutput shape mismatch");
        }

        Tensor gradInput({N, C, H, W}, 0.0);

        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t oh = 0; oh < OH; ++oh) {
                    for (size_t ow = 0; ow < OW; ++ow) {
                        const double go = gradOutput.at(n, c, oh, ow);

                        if (mode_ == PoolMode::Max) {
                            const size_t outIdx =
                                flatOutputIndex(n, c, oh, ow, C, OH, OW);

                            const size_t inputFlatIdx = maxIndices_[outIdx];

                            const size_t ih = inputFlatIdx / W;
                            const size_t iw = inputFlatIdx % W;

                            gradInput.at(n, c, ih, iw) += go;
                        } else {
                            size_t count = 0;

                            for (size_t kh = 0; kh < poolH_; ++kh) {
                                for (size_t kw = 0; kw < poolW_; ++kw) {
                                    const int ih = (oh * stride_ + kh) - padding_;
                                    const int iw = (ow * stride_ + kw) - padding_;

                                    if (inside(ih, iw, H, W)) {
                                        count++;
                                    }
                                }
                            }

                            if (count == 0) {
                                throw std::runtime_error("Pool4D average backward window has no valid cells");
                            }

                            const double share = go / static_cast<double>(count);

                            for (size_t kh = 0; kh < poolH_; ++kh) {
                                for (size_t kw = 0; kw < poolW_; ++kw) {
                                    const int ih = (oh * stride_ + kh) - padding_;
                                    const int iw = (ow * stride_ + kw) - padding_;

                                    if (inside(ih, iw, H, W)) {
                                        gradInput.at(n, c, ih, iw) += share;
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

};










} // namespace nn
