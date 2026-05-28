#pragma once

#include "nn/core/sequential.h"
#include "nn/layers/embedding_bag.h"
#include "nn/layers/linear.h"


namespace nn {

inline Sequential makeCBOWModel(size_t vocabSize,
                                size_t embeddingDim,
                                unsigned int seed = 42) {
    Sequential net;

    net.add<EmbeddingBag>(vocabSize, embeddingDim, seed);
    net.add<Linear>(embeddingDim, vocabSize, seed + 1);

    return net;
}


} // namespace nn
