#pragma once

#include "nn/core/module.h"
#include "nn/core/tensor.h"
#include "nn/core/parameter.h"
#include <random>
#include <stdexcept>


namespace nn {

class VanillaRNN : public Module {
private:
    size_t inputDim_;
    size_t hiddenDim_;

    Parameter Wx_; // [hiddenDim, inputDim]
    Parameter Wh_; // [hiddenDim, hiddenDim]
    Parameter b_;  // [hiddenDim]

    /** B=batch size, T=seq length, D=input dim, H=hidden dim */
    Tensor inputCache_;  // [B, T, D]
    Tensor hiddenCache_; // [B, T, H]

public:
    VanillaRNN(size_t inputDim,
               size_t hiddenDim,
               unsigned int seed = 42)
            : inputDim_(inputDim),
              hiddenDim_(hiddenDim),
              Wx_(std::vector<size_t>{hiddenDim, inputDim}),
              Wh_(std::vector<size_t>{hiddenDim, hiddenDim}),
              b_(std::vector<size_t>{hiddenDim}) {
        if (inputDim_ == 0 || hiddenDim_ == 0) {
            throw std::runtime_error("VanillaRNN dimensions must be positive");
        }

        std::mt19937 rng(seed);

        // Xavier uniform init for tanh.
        const double limitWx = std::sqrt(6.0 /
                                         static_cast<double>(inputDim_ + hiddenDim_));
        const double limitWh = std::sqrt(6.0 /
                                         static_cast<double>(hiddenDim_ + hiddenDim_));

        std::uniform_real_distribution<double> distWx(-limitWx, +limitWx);
        std::uniform_real_distribution<double> distWh(-limitWh, +limitWh);

        for (size_t i = 0; i < Wx_.value.numel(); i++) {
            Wx_.value[i] = distWx(rng);
        }

        for (size_t i = 0; i < Wh_.value.numel(); i++) {
            Wh_.value[i] = distWh(rng);
        }

        b_.value.fill(0.0);
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 3) {
            throw std::runtime_error("VanillaRNN expects input [B, T, D]");
        }

        const size_t B = x.shape()[0];
        const size_t T = x.shape()[1];
        const size_t D = x.shape()[2];

        if (D != inputDim_) {
            throw std::runtime_error("VanillaRNN input dimension mismatch");
        }

        inputCache_ = x;
        hiddenCache_ = Tensor({B, T, hiddenDim_}, 0.0);

        // Loop over sequences from the batch
        for (size_t n = 0; n < B; n++) {
            std::vector<double> hPrev(hiddenDim_, 0.0);

            /* Move through time. At each timestep, we compute a new hidden vector
            from current input x_t and prev hidden state h_{t-1} */
            for (size_t t = 0; t < T; t++) {
                std::vector<double> hCurr(hiddenDim_, 0.0);

                // Computes one coordinate of the hidden vector
                for (size_t h = 0; h < hiddenDim_; h++) {
                    double sum = b_.value[h];

                    /* Computes x_t * Wx
                    Dot product between: x[n, t, :] - current input vector, shape [D]
                    Wx[h, :] - weights from input to hidden unit h, shape [D] */
                    for (size_t d = 0; d < inputDim_; d++) {
                        sum += x.at(n, t, d) * Wx_.value.at(h, d);
                    }

                    /* Computes h_{t-1} * Wh
                    Dot product between: hPrev[:] - prev hidden vector, shape [H]
                    Wh[h, :] - weights from prev hidden to hidden unit h, shape [H] */
                    for (size_t hp = 0; hp < hiddenDim_; hp++) {
                        sum += hPrev[hp] * Wh_.value.at(h, hp);
                    }

                    hCurr[h] = std::tanh(sum);
                    hiddenCache_.at(n, t, h) = hCurr[h];
                }

                hPrev = hCurr;
            }
        }

        return hiddenCache_;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 3 || hiddenCache_.ndim() != 3) {
            throw std::runtime_error("VanillaRNN backward called before forward");
        }

        if (gradOutput.ndim() != 3) {
            throw std::runtime_error("VanillaRNN backward expects gradOutput [B, T, H]");
        }

        const size_t B = inputCache_.shape()[0];
        const size_t T = inputCache_.shape()[1];
        const size_t D = inputCache_.shape()[2];
        const size_t H = hiddenDim_;

        if (gradOutput.shape()[0] != B ||
            gradOutput.shape()[1] != T ||
            gradOutput.shape()[2] != H) {
            throw std::runtime_error("VanillaRNN gradOutput shape mismatch");
        }

        Tensor gradInput({B, T, D}, 0.0);

        /*
            Backpropagation through time.
            At each time t, hidden state h_t receives gradient
            from two places:
            1. gradOutput[n, t, :] - loss grad attached to output at this time
            2. dhNext - gradient flowing backward from h_{t+1}

        */
        for (size_t n = 0; n < B; n++) {
            std::vector<double> dhNext(H, 0.0);

            for (size_t tRev = 0; tRev < T; tRev++) {
                const size_t t = T - 1 - tRev;

                std::vector<double> dhPrev(H, 0.0);

                for (size_t h = 0; h < H; h++) {
                    const double h_t = hiddenCache_.at(n, t, h);

                    // dL/dh_t combines upstream gradient and future timestep.
                    const double dh = gradOutput.at(n, t, h) + dhNext[h];

                    /* Derivative of tanh. Contributes to four things:
                    bias, Wx, Wh, input vector at time t */
                    const double da = dh * (1.0 - h_t * h_t);

                    b_.grad[h] += da;

                    // Gradients wrt Wx and input x_t
                    for (size_t d = 0; d < D; d++) {
                        const double x_td = inputCache_.at(n, t, d);
                        Wx_.grad.at(h, d) += da * x_td;

                        gradInput.at(n, t, d) += da * Wx_.value.at(h, d);
                    }

                    // Previous hidden state h_{t-1}.
                    for (size_t hp = 0; hp < H; hp++) {
                        double hPrevValue = 0.0;

                        if (t > 0) {
                            hPrevValue = hiddenCache_.at(n, t-1, hp);
                        }

                        Wh_.grad.at(h, hp) += da * hPrevValue;

                        dhPrev[hp] += da * Wh_.value.at(h, hp);
                    }
                }

                dhNext = dhPrev;
            }
        }

        return gradInput;
    }

    std::vector<Parameter*> parameters() override {
        return {&Wx_, &Wh_, &b_};
    }

    void zeroGrad() override {
        Wx_.zeroGrad();
        Wh_.zeroGrad();
        b_.zeroGrad();
    }

    Parameter& Wx() { return Wx_; }
    Parameter& Wh() { return Wh_; }
    Parameter& bias() { return b_; }

    const Parameter& Wx() const { return Wx_; }
    const Parameter& Wh() const { return Wh_; }
    const Parameter& bias() const { return b_; }

    size_t inputDim() const { return inputDim_; }
    size_t hiddenDim() const { return hiddenDim_; }
};


} // namespace nn
