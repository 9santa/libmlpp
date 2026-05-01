#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "datasets/batch_utils.h"
#include "datasets/mnist_loader.h"

#include "models/neural_networks/architectures/lenet/lenet.h"
#include "models/neural_networks/losses/softmax_cross_entropy.h"
#include "models/neural_networks/optimizers/sgd.h"
#include "models/neural_networks/evalutaion/classification.h"

int main() {
    try {
        const std::string trainImagesPath = "data/mnist/train-images-idx3-ubyte";
        const std::string trainLabelsPath = "data/mnist/train-labels-idx1-ubyte";
        const std::string testImagesPath  = "data/mnist/t10k-images-idx3-ubyte";
        const std::string testLabelsPath  = "data/mnist/t10k-labels-idx1-ubyte";

        const size_t trainLimit = 5000;
        const size_t testLimit = 1000;

        std::cout << "Loading MNIST...\n";

        nn::Tensor trainX = nn::loadMNISTImages(trainImagesPath, trainLimit, true);
        nn::Tensor trainY = nn::loadMNISTLabels(trainLabelsPath, trainLimit);

        nn::Tensor testX = nn::loadMNISTImages(testImagesPath, testLimit, true);
        nn::Tensor testY = nn::loadMNISTLabels(testLabelsPath, testLimit);

        std::cout << "Train images: " << trainX.shape()[0] << "\n";
        std::cout << "Test images: " << testX.shape()[0] << "\n";

        nn::Sequential net = nn::makeLeNetMNIST(
            nn::LeNetActivation::ReLU,
            nn::PoolMode::Max
        );

        nn::SoftmaxCrossEntropy loss;
        nn::SGD optimizer(net.parameters(), 1e-3);

        const size_t batchSize = 32;
        const int epochs = 2;

        std::vector<size_t> indices(trainX.shape()[0]);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(42);

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);

            double totalLoss = 0.0;
            double totalAcc = 0.0;
            size_t batches = 0;

            for (size_t start = 0; start < indices.size(); start += batchSize) {
                nn::Tensor xBatch = nn::makeImageBatch(trainX, indices, start, batchSize);
                nn::Tensor yBatch = nn::makeLabelBatch(trainY, indices, start, batchSize);

                optimizer.zeroGrad();

                nn::Tensor logits = net.forward(xBatch);

                double L = loss.forward(logits, yBatch);
                nn::Tensor grad = loss.backward();

                net.backward(grad);

                optimizer.step();

                totalLoss += L;
                totalAcc += nn::batchAccuracy(logits, yBatch);
                ++batches;
            }

            std::cout << "Epoch " << epoch
                      << " train_loss=" << totalLoss / static_cast<double>(batches)
                      << " train_acc=" << totalAcc / static_cast<double>(batches)
                      << "\n";

            std::vector<size_t> testIndices(testX.shape()[0]);
            std::iota(testIndices.begin(), testIndices.end(), 0);

            double testAcc = 0.0;
            size_t testBatches = 0;

            for (size_t start = 0; start < testIndices.size(); start += batchSize) {
                nn::Tensor xBatch = nn::makeImageBatch(testX, testIndices, start, batchSize);
                nn::Tensor yBatch = nn::makeLabelBatch(testY, testIndices, start, batchSize);

                nn::Tensor logits = net.forward(xBatch);

                testAcc += nn::batchAccuracy(logits, yBatch);
                ++testBatches;
            }

            std::cout << "Epoch " << epoch
                      << " test_acc=" << testAcc / static_cast<double>(testBatches)
                      << "\n";
        }

        std::cout << "MNIST training finished.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
