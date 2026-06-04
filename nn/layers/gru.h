#pragma once

#include "nn/core/module.h"
#include "nn/core/tensor.h"
#include "nn/core/parameter.h"

#include <cmath>
#include <random>
#include <stdexcept>


namespace nn {

class GRU : public Module {
private:
    size_t inputDim_;
    size_t hiddenDim_;

    Parameter Wx_; // [3H, D]
    Parameter Wh_; // [3H, H]
    Parameter b_;  // [3H]

    Tensor inputCache_; // [B, T, D]

    Tensor zCache_; // [B, T, H]
    Tensor rCache_; // [B, T, H]
    Tensor gCache_; // [B, T, H]
    Tensor hCache_; // [B, T, H]

    size_t rowUpdate(size_t h) const {
        return h;
    }

    size_t rowReset(size_t h) const {
        return hiddenDim_ + h;
    }

    size_t rowCandidate(size_t h) const {
        return 2 * hiddenDim_ + h;
    }

    static double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

public:
    GRU(size_t inputDim,
        size_t hiddenDim,
        unsigned int seed = 42)
        : inputDim_(inputDim),
          hiddenDim_(hiddenDim),
          Wx_(std::vector<size_t>{3 * hiddenDim, inputDim}),
          Wh_(std::vector<size_t>{3 * hiddenDim, hiddenDim}),
          b_(std::vector<size_t>{3 * hiddenDim}) {
        if (inputDim_ == 0 || hiddenDim_ == 0) {
            throw std::runtime_error("GRU dimensions must be positive");
        }

        std::mt19937 rng(seed);

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
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 3) {
            throw std::runtime_error("GRU expects input [B, T, D]");
        }

        const size_t B = x.shape()[0];
        const size_t T = x.shape()[1];
        const size_t D = x.shape()[2];

        if (D != inputDim_) {
            throw std::runtime_error("GRU input dimension mismatch");
        }

        inputCache_ = x;

        zCache_ = Tensor({B, T, hiddenDim_}, 0.0);
        rCache_ = Tensor({B, T, hiddenDim_}, 0.0);
        gCache_ = Tensor({B, T, hiddenDim_}, 0.0);
        hCache_ = Tensor({B, T, hiddenDim_}, 0.0);

        for (size_t n = 0; n < B; n++) {
            std::vector<double> hPrev(hiddenDim_, 0.0);

            for (size_t t = 0; t < T; t++) {
                std::vector<double> z(hiddenDim_, 0.0);
                std::vector<double> r(hiddenDim_, 0.0);
                std::vector<double> g(hiddenDim_, 0.0);
                std::vector<double> hCurr(hiddenDim_, 0.0);

                // First compute update and reset gates.
                for (size_t h = 0; h < hiddenDim_; h++) {
                    double az = b_.value[rowUpdate(h)];
                    double ar = b_.value[rowReset(h)];

                    for (size_t d = 0; d < inputDim_; d++) {
                        const double x_td = x.at(n, t, d);

                        az += x_td * Wx_.value.at(rowUpdate(h), d);
                        ar += x_td * Wx_.value.at(rowReset(h), d);
                    }

                    for (size_t hp = 0; hp < hiddenDim_; hp++) {
                        const double h_prev = hPrev[hp];

                        az += h_prev * Wh_.value.at(rowUpdate(h), hp);
                        ar += h_prev * Wh_.value.at(rowReset(h), hp);
                    }

                    z[h] = sigmoid(az);
                    r[h] = sigmoid(ar);

                    zCache_.at(n, t, h) = z[h];
                    rCache_.at(n, t, h) = r[h];
                }

                // Compute candidate hidden state
                for (size_t h = 0; h < hiddenDim_; h++) {
                    double ag = b_.value[rowCandidate(h)];

                    for (size_t d = 0; d < inputDim_; d++) {
                        const double x_td = x.at(n, t, d);
                        ag += x_td * Wx_.value.at(rowCandidate(h), d);
                    }

                    for (size_t hp = 0; hp < hiddenDim_; hp++) {
                        const double resetHidden = r[hp] * hPrev[hp];
                        ag += resetHidden * Wh_.value.at(rowCandidate(h), hp);
                    }

                    g[h] = std::tanh(ag);
                    gCache_.at(n, t, h) = g[h];
                }

                // Final hidden state
                for (size_t h = 0; h < hiddenDim_; h++) {
                    hCurr[h] = z[h] * hPrev[h] + (1.0 - z[h]) * g[h];
                    hCache_.at(n, t, h) = hCurr[h];
                }

                hPrev = hCurr;
            }
        }

        return hCache_;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 3 || hCache_.ndim() != 3) {
            throw std::runtime_error("GRU backward called before forward");
        }

        if (gradOutput.ndim() != 3) {
            throw std::runtime_error("GRU backward expects gradOutput [B, T, H]");
        }

        const size_t B = inputCache_.shape()[0];
        const size_t T = inputCache_.shape()[1];
        const size_t D = inputCache_.shape()[2];
        const size_t H = hiddenDim_;

        if (gradOutput.shape()[0] != B ||
            gradOutput.shape()[1] != T ||
            gradOutput.shape()[2] != H) {
            throw std::runtime_error("GRU gradOutput shape mismatch");
        }

        Tensor gradInput({B, T, D}, 0.0);

        /*
            Backpropagation through time for GRU.

            h_t = z_t * h_prev + (1 - z_t) * g_t

            g_t = tanh(Wx_g x_t + Wh_g (r_t * h_prev) + b_g)

            z_t = sigmoid(Wx_z x_t + Wh_z h_prev + b_z)
            r_t = sigmoid(Wx_r x_t + Wh_r h_prev + b_r)
        */
        for (size_t n = 0; n < B; n++) {
            std::vector<double> dhNext(H, 0.0);

            for (size_t tRev = 0; tRev < T; tRev++) {
                const size_t t = T - 1 - tRev;

                std::vector<double> dhPrev(H, 0.0);

                std::vector<double> dzGate(H, 0.0);
                std::vector<double> drGate(H, 0.0);
                std::vector<double> dgGate(H, 0.0);

                std::vector<double> daz(H, 0.0);
                std::vector<double> dar(H, 0.0);
                std::vector<double> dag(H, 0.0);

                // Backward through:
                // h_t = z_t * h_prev + (1 - z_t) * g_t
                for (size_t h = 0; h < H; h++) {
                    const double z = zCache_.at(n, t, h);
                    const double g = gCache_.at(n, t, h);

                    const double hPrev =
                        (t > 0) ? hCache_.at(n, t-1, h) : 0.0;

                    const double dh =
                        gradOutput.at(n, t, h) + dhNext[h];

                    dzGate[h] += dh * (hPrev - g);
                    dgGate[h] += dh * (1.0 - z);

                    // Direct path fromn h_t to h_{t-1}
                    dhPrev[h] += dh * z;
                }

                // Backward through candidate:
                // g_t = tanh(a_g)
                for (size_t h = 0; h < H; h++) {
                    const double g = gCache_.at(n, t, h);
                    dag[h] = dgGate[h] * (1.0 - g * g);
                }

                /*
                    Candidate preactivation:

                    a_g[h] =
                        b_g[h]
                      + sum_d x[d] * Wx_g[h,d]
                      + sum_hp (r[hp] * hPrev[hp]) * Wh_g[h,hp]

                    This contributes gradients to:
                    - candidate weights
                    - input x_t
                    - reset gate r
                    - previous hidden state hPrev
                */
                for (size_t h = 0; h < H; h++) {
                    const size_t rowG = rowCandidate(h);
                    const double dact = dag[h];

                    b_.grad[rowG] += dact;

                    for (size_t d = 0; d < D; d++) {
                        const double x_td = inputCache_.at(n, t, d);

                        Wx_.grad.at(rowG, d) += dact * x_td;
                        gradInput.at(n, t, d) += dact * Wx_.value.at(rowG, d);
                    }

                    for (size_t hp = 0; hp < H; hp++) {
                        const double r = rCache_.at(n, t, hp);

                        const double hPrev =
                            (t > 0) ? hCache_.at(n, t-1, hp) : 0.0;

                        const double resetHidden = r * hPrev;

                        Wh_.grad.at(rowG, hp) += dact * resetHidden;

                        const double dResetHidden = dact * Wh_.value.at(rowG, hp);

                        drGate[hp] += dResetHidden * hPrev;
                        dhPrev[hp] += dResetHidden * r;
                    }
                }

                // Backward through update/reset sigmoid gates.
                for (size_t h = 0; h < H; h++) {
                    const double z = zCache_.at(n, t, h);
                    const double r = rCache_.at(n, t, h);

                    daz[h] = dzGate[h] * z * (1.0 - z);
                    dar[h] = drGate[h] * r * (1.0 - r);
                }

                // Backward through update gate affine transform.
                for (size_t h = 0; h < H; h++) {
                    const size_t rowZ = rowUpdate(h);
                    const double dact = daz[h];

                    b_.grad[h] += dact;

                    for (size_t d = 0; d < D; d++) {
                        const double x_td = inputCache_.at(n, t, d);

                        Wx_.grad.at(rowZ, d) += dact * x_td;
                        gradInput.at(n, t, d) += dact * Wx_.value.at(rowZ, d);
                    }

                    for (size_t hp = 0; hp < H; hp++) {
                        const double hPrev =
                            (t > 0) ? hCache_.at(n, t-1, hp) : 0.0;

                        Wh_.grad.at(rowZ, hp) += dact * hPrev;
                        dhPrev[hp] += dact * Wh_.value.at(rowZ, hp);
                    }
                }

                // Backward through reset gate affine transform.
                for (size_t h = 0; h < H; h++) {
                    const size_t rowR = rowReset(h);
                    const double dact = dar[h];

                    b_.grad[rowR] += dact;

                    for (size_t d = 0; d < D; d++) {
                        const double x_td = inputCache_.at(n, t, d);

                        Wx_.grad.at(rowR, d) += dact * x_td;
                        gradInput.at(n, t, d) += dact * Wx_.value.at(rowR, d);
                    }

                    for (size_t hp = 0; hp < H; hp++) {
                        const double hPrev =
                            (t > 0) ? hCache_.at(n, t-1, hp) : 0.0;

                        Wh_.grad.at(rowR, hp) += dact * hPrev;
                        dhPrev[hp] += dact * Wh_.value.at(rowR, hp);
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

    size_t inputDim() const {
        return inputDim_;
    }

    size_t hiddenDim() const {
        return hiddenDim_;
    }
};


} // namespace nn
