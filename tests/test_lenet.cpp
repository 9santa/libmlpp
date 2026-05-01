#include <iostream>

#include "lenet.h"
#include "../../losses/softmax_cross_entropy.h"
#include "../../optimizers/sgd.h"

int main() {
    nn::Sequential net = nn::makeLeNetMNIST(
        nn::LeNetActivation::Tanh,
        nn::PoolMode::Avg
    );

    const size_t batch = 4;

    nn::Tensor x({batch, 1, 28, 28}, 0.0);
    nn::Tensor y({batch}, 0.0);

    // Fake labels: 0, 1, 2, 3
    for (size_t i = 0; i < batch; i++) {
        y[i] = static_cast<double>(i % 10);
    }

    // Put some nonzero fake pixels
    for (size_t n = 0; n < batch; n++) {
        for (size_t h = 0; h < 28; h++) {
            for (size_t w = 0; w < 28; w++) {
                x.at(n, 0, h, w) = static_cast<double>((h + w + n) % 255) / 255.0;
            }
        }
    }

    nn::SoftmaxCrossEntropy loss;
    nn::SGD optimizer(net.parameters(), 0.01);
    optimizer.zeroGrad();

    nn::Tensor logits = net.forward(x);

    std::cout << "logits shape: [";
    for (size_t i = 0; i < logits.shape().size(); ++i) {
        std::cout << logits.shape()[i];
        if (i + 1 < logits.shape().size()) std::cout << ", ";
    }
    std::cout << "]\n";

    double L = loss.forward(logits, y);
    std::cout << "loss = " << L << "\n";

    nn::Tensor grad = loss.backward();
    net.backward(grad);

    optimizer.step();

    std::cout << "Forward/backward/step succeeded.\n";

    return 0;
}
