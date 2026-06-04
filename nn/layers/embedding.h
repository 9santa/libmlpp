#pragma once

#include "nn/core/module.h"
#include "nn/core/tensor.h"
#include "nn/core/parameter.h"

#include <cmath>
#include <random>
#include <stdexcept>


namespace nn {

class Embedding : public Module {
private:
    size_t vocabSize_;
    size_t embeddingDim_;

    Parameter E_;       // [vocabSize, embeddingDim]
    Tensor inputCache_; // [B, T], stores token ids as doubles

    size_t readId(double value) const {
        if (value < 0.0 || std::floor(value) != value) {
            throw std::runtime_error("Embedding input must contain non-negative integer ids");
        }

        size_t id = static_cast<size_t>(value);
        if (id >= vocabSize_) {
            throw std::runtime_error("Embedding token id out of range");
        }

        return id;
    }

public:
    Embedding(size_t vocabSize,
              size_t embeddingDim,
              unsigned int seed = 42)
        : vocabSize_(vocabSize),
          embeddingDim_(embeddingDim),
          E_(std::vector<size_t>{vocabSize, embeddingDim}) {
        if (vocabSize_ == 0) {
            throw std::runtime_error("Embedding vocabSize must be positive");
        }
        if (embeddingDim_ == 0) {
            throw std::runtime_error("Embedding embeddingDim must be positive");
        }

        std::mt19937 rng(seed);
        const double limit =
            std::sqrt(6.0 / static_cast<double>(vocabSize_ + embeddingDim_));
        std::uniform_real_distribution<double> dist(-limit, +limit);

        for (size_t i = 0; i < E_.value.numel(); i++) {
            E_.value[i] = dist(rng);
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 2) {
            throw std::runtime_error("Embedding expects input [B, T]");
        }

        const size_t B = x.shape()[0];
        const size_t T = x.shape()[1];

        inputCache_ = x;

        Tensor out({B, T, embeddingDim_}, 0.0);

        for (size_t n = 0; n < B; n++) {
            for (size_t t = 0; t < T; t++) {
                const size_t tokenId = readId(x.at(n, t));

                for (size_t d = 0; d < embeddingDim_; d++) {
                    out.at(n, t, d) = E_.value.at(tokenId, d);
                }
            }
        }

        return out;
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (inputCache_.ndim() != 2) {
            throw std::runtime_error("Embedding backward called before forward");
        }

        if (gradOutput.ndim() != 3) {
            throw std::runtime_error("Embedding backward expects gradOutput [B, T, D]");
        }

        const size_t B = inputCache_.shape()[0];
        const size_t T = inputCache_.shape()[1];

        if (gradOutput.shape()[0] != B ||
            gradOutput.shape()[1] != T ||
            gradOutput.shape()[2] != embeddingDim_) {
            throw std::runtime_error("Embedding gradOutput shape mismatch");
        }

        for (size_t n = 0; n < B; n++) {
            for (size_t t = 0; t < T; t++) {
                const size_t tokenId = readId(inputCache_.at(n, t));

                for (size_t d = 0; d < embeddingDim_; d++) {
                    E_.grad.at(tokenId, d) += gradOutput.at(n, t, d);
                }
            }
        }

        // No meaningful gradient wrt to token ids.
        return Tensor(inputCache_.shape(), 0.0);
    }

    std::vector<Parameter*> parameters() override {
        return {&E_};
    }

    void zeroGrad() override {
        E_.zeroGrad();
    }

    Parameter& embeddings() { return E_; }
    const Parameter& embeddings() const { return E_; }

    size_t vocabSize() const { return vocabSize_; }
    size_t embeddingDim() const { return embeddingDim_; }
};


} // namespace nn
