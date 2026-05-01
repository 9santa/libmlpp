#pragma once
#include "../core/module.h"
#include <limits>
#include <stdexcept>

namespace nn {

enum class PoolMode {
    Max,
    Avg
};

class Pool2D : public Module {
private:
    size_t poolH_;
    size_t poolW_;
    size_t stride_;
    size_t padding_;
    PoolMode mode_;

    Tensor inputCache_;

    // Used only for max pooling.
    // Stores flattened input index h * W + w for each output cell.
    std::vector<size_t> maxIndices_;

    static bool inside(size_t i, size_t j, size_t H, size_t W) {
        return i >= 0 && j >= 0 && i < H && j < W;
    }

public:
    Pool2D(size_t poolSize,
           size_t stride,
           PoolMode mode,
           size_t padding = 0)
        : poolH_(poolSize),
          poolW_(poolSize),
          stride_(stride),
          padding_(padding),
          mode_(mode) {

        if (poolSize == 0) {
            throw std::runtime_error("Pool2D pool size must be positive");
        }
        if (stride_ == 0) {
            throw std::runtime_error("Pool2D stride must be positive");
        }
    }

    explicit Pool2D(size_t poolSize, PoolMode mode = PoolMode::Max) : Pool2D(poolSize, poolSize, mode, 0) {}

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 2) {
            throw std::runtime_error("Pool2D expects 2D input [H, W]");
        }

        const size_t H = x.shape()[0], W = x.shape()[1];
        const size_t paddedH = H + 2 * padding_, paddedW = W + 2 * padding_;
        if (paddedH < H || paddedW < W) {
            throw std::runtime_error("Pool2D window larger than padded input");
        }

        const size_t outH = (paddedH - poolH_) / stride_ + 1;
        const size_t outW = (paddedW - poolW_) / stride_ + 1;

        inputCache_ = x;
        maxIndices_.assign(outH * outW, 0);

        Tensor y({outH, outW}, 0.0);

        for (size_t oh = 0; oh < outH; oh++) {
            for (size_t ow = 0; ow < outW; ow++) {
                if (mode_ == PoolMode::Max) {
                    double best = std::numeric_limits<double>::lowest();
                    size_t bestIndex = 0;
                    bool found = false;

                    for (size_t kh = 0; kh < poolH_; kh++) {
                        for (size_t kw = 0; kw < poolW_; kw++) {
                            int ih = (oh * stride_ + kh) - padding_;
                            int iw = (ow * stride_ + kw) - padding_;

                            if (inside(ih, iw, H, W)) {
                                double value = x.at(ih, iw);

                                if (!found || value > best) {
                                    found = true;
                                    best = value;
                                    bestIndex = ih * W + iw;
                                }
                            }
                        }
                    }

                    if (!found) {
                        throw std::runtime_error("Pool2D max window contains no valid cells");
                    }

                    y.at(oh, ow) = best;
                    maxIndices_[oh * outW + ow] = bestIndex;
                } else { // Avg pooling
                    double sum = 0.0;
                    size_t count = 0;

                    for (size_t kh = 0; kh < poolH_; kh++) {
                        for (size_t kw = 0; kw < poolW_; kw++) {
                            int ih = (oh * stride_ + kh) - padding_;
                            int iw = (ow * stride_ + kw) - padding_;

                            if (inside(ih, iw, H, W)) {
                                sum += x.at(ih, iw);
                                count++;
                            }
                        }
                    }

                    if (count == 0) {
                        throw std::runtime_error("Pool2D average window contains no valid cells");
                    }

                    y.at(oh, ow) = sum / static_cast<double>(count);
                }
            }
        }

        return y;
    }

    // Max pooling backward: sends gradient only to the location that won (maximum).
    // Avg pooling backward: distributes gradient evenly.
    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 2) {
            throw std::runtime_error("Pool2D backward called before forward");
        }

        const size_t H = inputCache_.shape()[0], W = inputCache_.shape()[1];
        const size_t paddedH = H + 2 * padding_, paddedW = W + 2 * padding_;
        const size_t outH = (paddedH - poolH_) / stride_ + 1;
        const size_t outW = (paddedW - poolW_) / stride_ + 1;

        if (gradOutput.ndim() != 2 || gradOutput.shape()[0] != outH || gradOutput.shape()[1] != outW) {
            throw std::runtime_error("Pool2D gradOutput shape mismatch");
        }

        Tensor gradInput({H, W}, 0.0);

        for (size_t oh = 0; oh < outH; oh++) {
            for (size_t ow = 0; ow < outW; ow++) {
                const double go = gradOutput.at(oh, ow);

                if (mode_ == PoolMode::Max) {
                    size_t idx = maxIndices_[oh * outW + ow];
                    size_t ih = idx / W;
                    size_t iw = idx % W;

                    gradInput.at(ih, iw) += go;
                } else {
                    size_t count = 0;

                    for (size_t kh = 0; kh < poolH_; kh++) {
                        for (size_t kw = 0; kw < poolW_; kw++) {
                            int ih = (oh * stride_ + kh) - padding_;
                            int iw = (ow * stride_ + kw) - padding_;

                            if (inside(ih, iw, H, W)) count++;
                        }
                    }

                    if (count == 0) {
                        throw std::runtime_error("Pool2D average backward window has no valid cells");
                    }

                    const double share = go / static_cast<double>(count);
                    // Distribute grad
                    for (size_t kh = 0; kh < poolH_; kh++) {
                        for (size_t kw = 0; kw < poolW_; kw++) {
                            int ih = (oh * stride_ + kh) - padding_;
                            int iw = (ow * stride_ + kw) - padding_;

                            if (inside(ih, iw, H, W)) {
                                gradInput.at(ih, iw) += share;
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
