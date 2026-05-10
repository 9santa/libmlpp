#pragma once

#include "nn/core/tensor.h"
#include <stdexcept>

namespace nn {

inline Tensor makeImageBatch(const Tensor& images,
                             const std::vector<size_t>& indices,
                             size_t start,
                             size_t batchSize) {
    if (images.ndim() != 4) {
        throw std::runtime_error("makeImageBatch expects images [N, C, H, W]");
    }

    const auto& s = images.shape();

    const size_t N = s[0];
    const size_t C = s[1];
    const size_t H = s[2];
    const size_t W = s[3];

    if (start >= indices.size()) {
        throw std::runtime_error("Batch start out of range");
    }

    const size_t actualBatch = std::min(batchSize, indices.size() - start);

    Tensor batch({actualBatch, C, H, W}, 0.0);

    for (size_t bi = 0; bi < actualBatch; bi++) {
        const size_t n = indices[start + bi];

        if (n >= N) {
            throw std::runtime_error("Image batch index out of range");
        }

        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    batch.at(bi, c, h, w) = images.at(n, c, h, w);
                }
            }
        }
    }

    return batch;
}

inline Tensor makeLabelBatch(const Tensor& labels,
                             const std::vector<size_t>& indices,
                             size_t start,
                             size_t batchSize) {
    if (labels.ndim() != 1) {
        throw std::runtime_error("makeLabelBatch expects labels [N]");
    }

    const size_t N = labels.shape()[0];

    if (start >= indices.size()) {
        throw std::runtime_error("Batch start out of range");
    }

    const size_t actualBatch = std::min(batchSize, indices.size() - start);

    Tensor batch({actualBatch}, 0.0);

    for (size_t bi = 0; bi < actualBatch; ++bi) {
        const size_t n = indices[start + bi];

        if (n >= N) {
            throw std::runtime_error("Label batch index out of range");
        }

        batch[bi] = labels[n];
    }

    return batch;
}

} // namespace nn
