#pragma once

#include "nn/core/module.h"
#include "nn/core/tensor.h"
#include "nn/core/parameter.h"
#include <random>
#include <stdexcept>


namespace nn {

class EmbeddingBag : public Module {
private:
    size_t vocabSize_;
    size_t embeddingDim_;   // maybe 300?

    Parameter E_;           // [vocabSize, embeddingDim]
    Tensor inputCache_;     // [batch, contextSize], stores word ids as doubles

    // Helper func: read & validate double id to size_t
    size_t readId(double value) const {
        if (value < 0.0 || std::floor(value) != value) {
            throw std::runtime_error("EmbeddingBag input must contain non-negative integer ids");
        }

        size_t id = static_cast<size_t>(value);
        if (id >= vocabSize_) {
            throw std::runtime_error("EmbeddingBag word id out of range");
        }
        return id;
    }


public:
    EmbeddingBag(size_t vocabSize,
                 size_t embeddingDim,
                 unsigned int seed = 42)
        : vocabSize_(vocabSize),
          embeddingDim_(embeddingDim),
          E_(std::vector<size_t>{vocabSize, embeddingDim}) {
        if (vocabSize_ == 0) {
            throw std::runtime_error("EmbeddingBag vocabSize must be positive");
        }
        if (embeddingDim == 0) {
            throw std::runtime_error("EmbeddingBag embeddingDim must be positive");
        }

        // Random parameter initialization
        std::mt19937 rng(seed);
        double limit = 2.0 * static_cast<double>(embeddingDim_);
        std::uniform_real_distribution<double> dist(-limit, +limit);

        for (size_t i = 0; i < E_.value.numel(); i++) {
            E_.value[i] = dist(rng);
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 2) {
            throw std::runtime_error("EmbeddingBag expects input [batch, contextSize]");
        }

        const size_t batchSize = x.shape()[0];
        const size_t contextSize = x.shape()[1];

        if (contextSize == 0) {
            throw std::runtime_error("EmbeddingBag contextSize must be positive");
        }

        inputCache_ = x;

        Tensor out({batchSize, embeddingDim_}, 0.0);

        for (size_t n = 0; n < batchSize; n++) {
            for (size_t j = 0; j < contextSize; j++) {
                size_t wordId = readId(x.at(n, j));

                for (size_t d = 0; d < embeddingDim_; d++) {
                    out.at(n, d) += E_.value.at(wordId, d);
                }
            }
        }

        return out;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 2) {
            throw std::runtime_error("EmbeddingBag backward called before forward");
        }

        if (gradOutput.ndim() != 2) {
            throw std::runtime_error("EmbeddingBag backward expects gradOutput [batchSize, embeddingDim]");
        }

        const size_t batchSize = inputCache_.shape()[0];
        const size_t contextSize = inputCache_.shape()[1];

        if (gradOutput.shape()[0] != batchSize || gradOutput.shape()[1] != embeddingDim_) {
            throw std::runtime_error("EmbeddingBag gradOutput shape mismatch");
        }

        double scale = 1.0;

        for (size_t n = 0; n < batchSize; n++) {
            for (size_t j = 0; j < contextSize; j++) {
                size_t wordId = readId(inputCache_.at(n, j));

                for (size_t d = 0; d < embeddingDim_; d++) {
                    E_.grad.at(wordId, d) += gradOutput.at(n, d) * scale;
                }
            }
        }

        // Word ids are discrete, so there is no meaningful gradient wrt input ids.
        return Tensor(inputCache_.shape(), 0.0);
    }

    std::vector<Parameter*> parameters() override {
        return {&E_};
    }

    void zeroGrad() override { E_.zeroGrad(); }

    Parameter& embeddings() { return E_; }

    const Parameter& embeddings() const { return E_; }
};


} // namespace nn
