#include "../core/tensor.h"
#include "../core/module.h"
#include "../core/parameter.h"
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nn {

// 2D cross-correlation
// X: [H, W]
// K: [KH, KW]
// Y: [H - KH + 1, W - KW + 1]
inline Tensor corr2d(const Tensor& X, const Tensor& K) {
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
            Y.at(i, j) = sum;
        }
    }

    return Y;
}

class Conv2D : public Module {
private:
    Parameter W_;       // [KH, KW]
    Parameter b_;       // [1]
    Tensor inputCache_; // [H, W]

public:
    explicit Conv2D(size_t kernel_size, unsigned int seed = 42)
        : W_(std::vector<size_t>{kernel_size, kernel_size}), b_(std::vector<size_t>{1}) {

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
            throw std::runtime_error("Conv2D currently expects 2D input [H, W]");
        }

        inputCache_ = x;
        Tensor y = corr2d(x, W_.value);
        // add scalar bias
        for (size_t i = 0; i < y.numel(); i++) {
            y[i] += b_.value[0];
        }

        return y;
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
