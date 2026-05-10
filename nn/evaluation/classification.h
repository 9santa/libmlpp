#pragma once

#include "../core/module.h"
#include "../core/tensor.h"

#include <stdexcept>

namespace nn {

inline size_t argmaxRow(const Tensor& logits, size_t row) {
    if (logits.ndim() != 2) {
        throw std::runtime_error("argmaxRow expects logits [N, classes]");
    }

    const size_t classes = logits.shape()[1];

    size_t best = 0;
    double bestValue = logits.at(row, 0);

    for (size_t c = 0; c < classes; c++) {
        double v = logits.at(row, c);
        if (v > bestValue) {
            bestValue = v;
            best = c;
        }
    }

    return best;
}

inline double batchAccuracy(const Tensor& logits, const Tensor& labels) {
    if (logits.ndim() != 2) {
        throw std::runtime_error("batchAccuracy expects logits [N, classes]");
    }
    if (labels.ndim() != 1) {
        throw std::runtime_error("batchAccuracy expects labels [N]");
    }
    if (logits.shape()[0] != labels.shape()[0]) {
        throw std::runtime_error("batchAccuracy batch size mismatch");
    }

    const size_t N = logits.shape()[0];

    size_t correct = 0;
    for (size_t n = 0; n < N; n++) {
        size_t pred = argmaxRow(logits, n);
        size_t truth = static_cast<size_t>(labels[n]);

        if (pred == truth) correct++;
    }

    return static_cast<double>(correct) / static_cast<double>(N);
}

} // namespace nn
