#include "datasets/text/vocab.h"
#include "datasets/text/cbow_dataset.h"

#include "nn/architectures/cbow/cbow.h"
#include "nn/core/sequential.h"
#include "nn/core/tensor.h"
#include "nn/losses/softmax_cross_entropy.h"
#include "nn/optimizers/sgd.h"
#include <exception>
#include <iostream>
#include <random>
#include <algorithm>


/*
context ids [B, 2W]
      ↓
EmbeddingBag
      ↓
total (sum) context embedding [B, D]
      ↓
Linear
      ↓
logits over vocabulary [B, V]
      ↓
SoftmaxCrossEntropy(target center word)
*/


int main() {
    try {
        std::string corpus = 
            "the quick brown fox jumps over the lazy dog "
            "the dog sleeps in the warm sun "
            "the fox runs through the forest "
            "the quick dog runs over the hill "
            "the brown fox is quick and the dog is lazy";

        std::vector<std::string> tokens = nn::simpleTokenize(corpus);

        nn::Vocab vocab = nn::buildVocab(tokens);
        std::vector<size_t> ids = nn::encodeTokens(tokens, vocab);

        const size_t windowSize = 2;
        auto examples = nn::makeCBOWExamples(ids, windowSize);

        std::cout << "tokens: " << tokens.size() << "\n";
        std::cout << "vocab size: " << vocab.size() << "\n";
        std::cout << "cbow examples: " << examples.size() << "\n";

        const size_t embeddingDim = 16;
        const size_t batchSize = 8;
        const int epochs = 2000;

        nn::Sequential net = nn::makeCBOWModel(vocab.size(), embeddingDim);

        nn::SoftmaxCrossEntropy loss;
        nn::SGD optimizer(net.parameters(), 1e-2);

        std::vector<size_t> indices(examples.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(42);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            std::shuffle(indices.begin(), indices.end(), rng);

            double total_loss = 0.0;
            size_t batches = 0;

            for (size_t start = 0; start < indices.size(); start += batchSize) {
                nn::Tensor xBatch =
                    nn::makeCBOWInputBatch(examples, indices, start, batchSize);

                nn::Tensor yBatch =
                    nn::makeCBOWTargetBatch(examples, indices, start, batchSize);

                optimizer.zeroGrad();

                nn::Tensor logits = net.forward(xBatch);

                double L = loss.forward(logits, yBatch);
                nn::Tensor grad = loss.backward();

                net.backward(grad);
                optimizer.step();

                total_loss += L;
                ++batches;
            }

            if (epoch % 20 == 0) {
                std::cout << "epoch " << epoch
                          << " loss=" << total_loss / static_cast<double>(batches)
                          << "\n";
            }
        }

        std::cout << "CBOW training finished.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error:" << e.what() << "\n";
        return 1;
    }

    return 0;
}
