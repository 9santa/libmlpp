#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nn/layers/embedding.h"
#include "nn/layers/embedding_bag.h"


namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, const char* message) {
    if (std::fabs(actual - expected) > 1e-12) {
        throw std::runtime_error(message);
    }
}

void fillEmbeddings(nn::Parameter& E) {
    for (size_t token = 0; token < E.value.shape()[0]; token++) {
        for (size_t d = 0; d < E.value.shape()[1]; d++) {
            E.value.at(token, d) =
                10.0 * static_cast<double>(token) + static_cast<double>(d);
        }
    }
}

void testEmbedding() {
    nn::Embedding embedding(5, 3);
    fillEmbeddings(embedding.embeddings());

    nn::Tensor x({2, 3}, 0.0);
    x.at(0, 0) = 0.0;
    x.at(0, 1) = 2.0;
    x.at(0, 2) = 2.0;
    x.at(1, 0) = 4.0;
    x.at(1, 1) = 1.0;
    x.at(1, 2) = 0.0;

    nn::Tensor y = embedding.forward(x);

    require(y.shape() == std::vector<size_t>({2, 3, 3}),
            "Embedding output shape mismatch");
    requireNear(y.at(0, 1, 2), 22.0, "Embedding lookup value mismatch");
    requireNear(y.at(1, 0, 1), 41.0, "Embedding lookup value mismatch");

    embedding.zeroGrad();
    nn::Tensor gy(y.shape(), 1.0);
    nn::Tensor gx = embedding.backward(gy);

    require(gx.shape() == std::vector<size_t>({2, 3}),
            "Embedding gradInput shape mismatch");
    requireNear(gx.at(0, 0), 0.0, "Embedding gradInput should be zero");
    requireNear(embedding.embeddings().grad.at(0, 0), 2.0,
                "Embedding repeated id gradient mismatch");
    requireNear(embedding.embeddings().grad.at(2, 1), 2.0,
                "Embedding repeated id gradient mismatch");
    requireNear(embedding.embeddings().grad.at(3, 2), 0.0,
                "Embedding unused id gradient mismatch");
}

void testEmbeddingBagSum() {
    nn::EmbeddingBag bag(5, 2);
    fillEmbeddings(bag.embeddings());

    nn::Tensor x({1, 3}, 0.0);
    x.at(0, 0) = 0.0;
    x.at(0, 1) = 1.0;
    x.at(0, 2) = 1.0;

    nn::Tensor y = bag.forward(x);

    require(y.shape() == std::vector<size_t>({1, 2}),
            "EmbeddingBag sum output shape mismatch");
    requireNear(y.at(0, 0), 20.0, "EmbeddingBag sum value mismatch");
    requireNear(y.at(0, 1), 23.0, "EmbeddingBag sum value mismatch");

    bag.zeroGrad();
    nn::Tensor gy({1, 2}, 1.0);
    bag.backward(gy);

    requireNear(bag.embeddings().grad.at(1, 0), 2.0,
                "EmbeddingBag sum repeated id gradient mismatch");
}

void testEmbeddingBagMean() {
    nn::EmbeddingBag bag(5, 2, nn::EmbeddingBagMode::Mean);
    fillEmbeddings(bag.embeddings());

    nn::Tensor x({1, 2}, 0.0);
    x.at(0, 0) = 1.0;
    x.at(0, 1) = 3.0;

    nn::Tensor y = bag.forward(x);

    require(y.shape() == std::vector<size_t>({1, 2}),
            "EmbeddingBag mean output shape mismatch");
    requireNear(y.at(0, 0), 20.0, "EmbeddingBag mean value mismatch");
    requireNear(y.at(0, 1), 21.0, "EmbeddingBag mean value mismatch");

    bag.zeroGrad();
    nn::Tensor gy({1, 2}, 1.0);
    bag.backward(gy);

    requireNear(bag.embeddings().grad.at(1, 0), 0.5,
                "EmbeddingBag mean gradient mismatch");
    requireNear(bag.embeddings().grad.at(3, 1), 0.5,
                "EmbeddingBag mean gradient mismatch");
}

} // namespace


int main() {
    testEmbedding();
    testEmbeddingBagSum();
    testEmbeddingBagMean();

    std::cout << "Embedding tests succeeded.\n";
    return 0;
}
