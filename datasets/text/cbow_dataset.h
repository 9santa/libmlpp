#pragma once

#include "nn/core/tensor.h"
#include <stdexcept>


namespace nn {

struct CBOWExample {
    std::vector<size_t> context;
    size_t target = 0;
};

inline std::vector<CBOWExample> makeCBOWExamples(
    const std::vector<size_t>& tokenIds,
    size_t windowSize
) {
    if (windowSize == 0) {
        throw std::runtime_error("CBOW windowSize must be positive");
    }

    std::vector<CBOWExample> examples;

    if (tokenIds.size() < 2 * windowSize + 1) {
        return examples;
    }

    for (size_t center = windowSize; center + windowSize < tokenIds.size(); center++) {
        CBOWExample ex;
        ex.context.resize(2 * windowSize);

        for (size_t offset = windowSize; offset > 0; offset--) {
            ex.context.push_back(tokenIds[center - offset]);
        }

        for (size_t offset = 1; offset <= windowSize; offset++) {
            ex.context.push_back(tokenIds[center + offset]);
        }

        ex.target = tokenIds[center];
        examples.push_back(std::move(ex));
    }

    return examples;
}


inline Tensor makeCBOWInputBatch(const std::vector<CBOWExample>& examples,
                                 const std::vector<size_t>& indices,
                                 size_t start,
                                 size_t batchSize) {
    if (start >= indices.size()) {
        throw std::runtime_error("CBOW batch start out of range");
    }

    const size_t actualBatch = std::min(batchSize, indices.size() - start);

    if (actualBatch == 0) {
        throw std::runtime_error("CBOW empty batch");
    }

    const size_t contextSize = examples[indices[start]].context.size();

    Tensor x({actualBatch, contextSize}, 0.0);

    for (size_t bi = 0; bi < actualBatch; bi++) {
        const size_t exIdx = indices[start + bi];

        if (exIdx >= examples.size()) {
            throw std::runtime_error("CBOW example index out of range");
        }

        const auto& context = examples[exIdx].context;

        if (context.size() != contextSize) {
            throw std::runtime_error("CBOW inconsistent context size");
        }

        for (size_t j = 0; j < contextSize; j++) {
            x.at(bi, j) = static_cast<double>(context[j]);
        }
    }

    return x;
}


inline Tensor makeCBOWTargetBatch(const std::vector<CBOWExample>& examples,
                                  const std::vector<size_t>& indices,
                                  size_t start,
                                  size_t batchSize) {
    if (start >= indices.size()) {
        throw std::runtime_error("CBOW target batch start out of range");
    }

    const size_t actualBatch = std::min(batchSize, indices.size() - start);

    Tensor y({actualBatch}, 0.0);

    for (size_t bi = 0; bi < actualBatch; bi++) {
        const size_t exIdx = indices[start + bi];

        if (exIdx >= examples.size()) {
            throw std::runtime_error("CBOW example index out of range");
        }

        y[bi] = static_cast<double>(examples[exIdx].target);
    }

    return y;
}


} // namespace nn
