#pragma once

#include "nn/core/tensor.h"

#include <stdexcept>

namespace nn {

/** Adds two tensors element-wise. */
inline Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("add: tensor shape mismatch");
    }

    Tensor out = a;
    for (size_t i = 0; i < out.numel(); i++) {
        out[i] += b[i];
    }

    return out;
}

/** Reverses the time dimension. Used in RNN. */
inline Tensor reverseTime(const Tensor& x) {
    if (x.ndim() != 3) {
        throw std::runtime_error("reverseTime expects tensor [B, T, D]");
    }

    const size_t B = x.shape()[0];
    const size_t T = x.shape()[1];
    const size_t D = x.shape()[2];

    Tensor out({B, T, D}, 0.0);

    for (size_t b = 0; b < B; b++) {
        for (size_t t = 0; t < T; t++) {
            const size_t rt = T - 1 - t;

            for (size_t d = 0; d < D; d++) {
                out.at(b, t, d) = x.at(b, rt, d);
            }
        }
    }

    return out;
}

inline Tensor concatLastDim3D(const Tensor& a, const Tensor& b) {
    if (a.ndim() !=3 || b.ndim() != 3) {
        throw std::runtime_error("concatLastDim3D expects 3D tensors");
    }

    if (a.shape()[0] != b.shape()[0] ||
        a.shape()[1] != b.shape()[1]) {
        throw std::runtime_error("concatLastDim3D batch/time dimension mismatch");
    }

    const size_t B = a.shape()[0];
    const size_t T = a.shape()[1];
    const size_t D1 = a.shape()[2];
    const size_t D2 = b.shape()[2];

    Tensor out({B, T, D1 + D2}, 0.0);

    for (size_t n = 0; n < B; n++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t d = 0; d < D1; d++) {
                out.at(n, t, d) = a.at(n, t, d);
            }

            for (size_t d = 0; d < D2; d++) {
                out.at(n, t, D1 + d) = b.at(n, t, d);
            }
        }
    }

    return out;
}

inline std::pair<Tensor, Tensor> splitLastDim3D(const Tensor& x) {
    if (x.ndim() != 3) {
        throw std::runtime_error("splitLastDim3D expects tensor [B, T, D]");
    }

    const size_t B = x.shape()[0];
    const size_t T = x.shape()[1];
    const size_t D = x.shape()[2];

    if (D & 1) {
        throw std::runtime_error("splitLastDim3D expects even last dimension");
    }

    const size_t H = D / 2;

    Tensor left({B, T, H}, 0.0);
    Tensor right({B, T, H}, 0.0);

    for (size_t n = 0; n < B; n++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t h = 0; h < H; h++) {
                left.at(n, t, h) = x.at(n, t, h);
                right.at(n, t, h) = x.at(n, t, H + h);
            }
        }
    }

    return {left, right};
}


} // namespace nn
