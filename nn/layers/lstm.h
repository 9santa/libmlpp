#pragma once

#include "nn/core/module.h"
#include "nn/core/tensor.h"
#include "nn/core/parameter.h"

#include <cmath>
#include <random>
#include <stdexcept>


namespace nn {

class LSTM : public Module {
private:
    size_t inputDim_;
    size_t hiddenDim_;


    /*
        We will store all four gates in one combined parameter matrix:
        Wx: [4H, D]
        Wh: [4H, H]
        b:  [4H]

        Gate layout:
        0H..1H = input gate i
        1H..2H = forget gate f
        2H..3H = candidate gate g
        3H..4H = output gate o
    */
    Parameter Wx_; // [4H, D]
    Parameter Wh_; // [4H, H]
    Parameter b_;  // [4H]

    Tensor inputCache_; // [B, T, D]

    Tensor iCache_; // [B, T, H]
    Tensor fCache_; // [B, T, H]
    Tensor gCache_; // [B, T, H]
    Tensor oCache_; // [B, T, H]

    Tensor cCache_; // [B, T, H]
    Tensor hCache_; // [B, T, H]

    const size_t rowInput(size_t h) const {
        return h;
    }

    size_t rowForget(size_t h) const {
        return hiddenDim_ + h;
    }

    size_t rowCandidate(size_t h) const {
        return 2 * hiddenDim_ + h;
    }

    size_t rowOutput(size_t h) const {
        return 3 * hiddenDim_ + h;
    }

    static double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

public:
    LSTM(size_t inputDim,
         size_t hiddenDim,
         unsigned int seed = 42)
        : inputDim_(inputDim),
          hiddenDim_(hiddenDim),
          Wx_(std::vector<size_t>{4 * hiddenDim, inputDim}),
          Wh_(std::vector<size_t>{4 * hiddenDim, hiddenDim}),
          b_(std::vector<size_t>{4 * hiddenDim}) {
        if (inputDim_ == 0 || hiddenDim_ == 0) {
            throw std::runtime_error("LSTM dimensions must be positive");
        }

        std::mt19937 rng(seed);

        // Xavier uniform init for tanh.
        const double limitWx =
            std::sqrt(6.0 / static_cast<double>(inputDim_ + hiddenDim_));

        const double limitWh =
            std::sqrt(6.0 / static_cast<double>(hiddenDim_ + hiddenDim_));

        std::uniform_real_distribution<double> distWx(-limitWx, +limitWx);
        std::uniform_real_distribution<double> distWh(-limitWh, +limitWh);

        for (size_t i = 0; i < Wx_.value.numel(); i++) {
            Wx_.value[i] = distWx(rng);
        }

        for (size_t i = 0; i < Wh_.value.numel(); i++) {
            Wh_.value[i] = distWh(rng);
        }

        b_.value.fill(0.0);

        /* Common LSTM trick:
        initialize forget gate bias to 1 so the network initially
        prefers remembering rather than immideately forgetting */
        for (size_t h = 0; h < hiddenDim_; h++) {
            b_.value[rowForget(h)] = 1.0;
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 3) {
            throw std::runtime_error("LSTM expects input [B, T, D]");
        }

        const size_t B = x.shape()[0];
        const size_t T = x.shape()[1];
        const size_t D = x.shape()[2];

        if (D != inputDim_) {
            throw std::runtime_error("LSTM input dimension mismatch");
        }

        inputCache_ = x;

        iCache_ = Tensor({B, T, hiddenDim_}, 0.0);
        fCache_ = Tensor({B, T, hiddenDim_}, 0.0);
        gCache_ = Tensor({B, T, hiddenDim_}, 0.0);
        oCache_ = Tensor({B, T, hiddenDim_}, 0.0);

        cCache_ = Tensor({B, T, hiddenDim_}, 0.0);
        hCache_ = Tensor({B, T, hiddenDim_}, 0.0);

        for (size_t n = 0; n < B; n++) {
            std::vector<double> hPrev(hiddenDim_, 0.0);
            std::vector<double> cPrev(hiddenDim_, 0.0);

            for (size_t t = 0; t < T; t++) {
                std::vector<double> hCurr(hiddenDim_, 0.0);
                std::vector<double> cCurr(hiddenDim_, 0.0);

                for (size_t h = 0; h < hiddenDim_; h++) {
                    double ai = b_.value[rowInput(h)];
                    double af = b_.value[rowForget(h)];
                    double ag = b_.value[rowCandidate(h)];
                    double ao = b_.value[rowOutput(h)];

                    for (size_t d = 0; d < inputDim_; d++) {
                        const double x_td = x.at(n, t, d);

                        ai += x_td * Wx_.value.at(rowInput(h), d);
                        af += x_td * Wx_.value.at(rowForget(h), d);
                        ag += x_td * Wx_.value.at(rowCandidate(h), d);
                        ao += x_td * Wx_.value.at(rowOutput(h), d);
                    }

                    for (size_t hp = 0; hp < hiddenDim_; hp++) {
                        const double h_prev = hPrev[hp];

                        ai += h_prev * Wh_.value.at(rowInput(h), hp);
                        af += h_prev * Wh_.value.at(rowForget(h), hp);
                        ag += h_prev * Wh_.value.at(rowCandidate(h), hp);
                        ao += h_prev * Wh_.value.at(rowOutput(h), hp);
                    }

                    const double i = sigmoid(ai);
                    const double f = sigmoid(af);
                    const double g = std::tanh(ag);
                    const double o = sigmoid(ao);

                    const double c = f * cPrev[h] + i * g;
                    const double hValue = o * std::tanh(c);

                    iCache_.at(n, t, h) = i;
                    fCache_.at(n, t, h) = f;
                    gCache_.at(n, t, h) = g;
                    oCache_.at(n, t, h) = o;

                    cCache_.at(n, t, h) = c;
                    hCache_.at(n, t, h) = hValue;

                    cCurr[h] = c;
                    hCurr[h] = hValue;
                }

                cPrev = cCurr;
                hPrev = hCurr;
            }
        }

        return hCache_;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 3 || hCache_.ndim() != 3 || cCache_.ndim() != 3) {
            throw std::runtime_error("LSTM backward called before forward");
        }

        if (gradOutput.ndim() != 3) {
            throw std::runtime_error("LSTM backward expects gradOutput [B, T, H]");
        }

        const size_t B = inputCache_.shape()[0];
        const size_t T = inputCache_.shape()[1];
        const size_t D = inputCache_.shape()[2];
        const size_t H = hiddenDim_;

        if (gradOutput.shape()[0] != B ||
            gradOutput.shape()[1] != T ||
            gradOutput.shape()[2] != H) {
            throw std::runtime_error("LSTM gradOutput shape mismatch");
        }

        Tensor gradInput({B, T, D}, 0.0);

        /*
            Backpropagation through time for LSTM.

            h_t = o_t * tanh(c_t)
            c_t = f_t * c_{t-1} + i_t * g_t

            We carry two future gradients:

            dhNext = gradient flowing from h_{t+1}
            dcNext = gradient flowing through cell state c_{t+1}
        */
        for (size_t n = 0; n < B; n++) {
            std::vector<double> dhNext(H, 0.0);
            std::vector<double> dcNext(H, 0.0);

            for (size_t tRev = 0; tRev < T; tRev++) {
                const size_t t = T - 1 - tRev;

                std::vector<double> dhPrev(H, 0.0);
                std::vector<double> dcPrev(H, 0.0);

                // Combined gate pre-activation gradients:
                // da = [dai, daf, dag, dao]
                std::vector<double> da(4 * H, 0.0);

                for (size_t h = 0; h < H; h++) {
                    const double i = iCache_.at(n, t, h);
                    const double f = fCache_.at(n, t, h);
                    const double g = gCache_.at(n, t, h);
                    const double o = oCache_.at(n, t, h);

                    const double c = cCache_.at(n, t, h);

                    const double cPrev =
                        (t > 0) ? cCache_.at(n, t-1, h) : 0.0;

                    const double tanhC = std::tanh(c);

                    // h_t receives gradient from output + future timestep t+1
                    const double dh =
                        gradOutput.at(n, t, h) + dhNext[h];

                    // h_t = o_t * tanh(c_t)
                    const double doGate = dh * tanhC;

                    // Gradient into c_t from h_t + future cell state
                    const double dc =
                        dh * o * (1.0 - tanhC * tanhC) + dcNext[h];

                    // c_t = f_t * c_{t-1} + i_t * g_t
                    const double dfGate = dc * cPrev;
                    const double diGate = dc * g;
                    const double dgGate = dc * i;

                    dcPrev[h] += dc * f;

                    // Gate activation derivatives
                    da[rowInput(h)] = diGate * i * (1.0 - i);
                    da[rowForget(h)] = dfGate * f * (1.0 - f);
                    da[rowCandidate(h)] = dgGate * (1.0 - g * g);
                    da[rowOutput(h)] = doGate * o * (1.0 - o);
                }

                for (size_t r = 0; r < 4 * H; r++) {
                    const double dact = da[r];

                    b_.grad[r] += dact;

                    for (size_t d = 0; d < D; d++) {
                        const double x_td = inputCache_.at(n, t, d);

                        Wx_.grad.at(r, d) += dact * x_td;

                        gradInput.at(n, t, d) += dact * Wx_.value.at(r, d);
                    }

                    for (size_t hp = 0; hp < H; hp++) {
                        const double hPrev =
                            (t > 0) ? hCache_.at(n, t-1, hp) : 0.0;

                        Wh_.grad.at(r, hp) += dact * hPrev;

                        dhPrev[hp] += dact * Wh_.value.at(r, hp);
                    }
                }

                dhNext = dhPrev;
                dcNext = dcPrev;
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

    size_t inputDim() const {
        return inputDim_;
    }

    size_t hiddenDim() const {
        return hiddenDim_;
    }
};


} // namespace nn
