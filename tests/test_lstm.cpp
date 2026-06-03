#include <iostream>

#include "nn/layers/lstm.h"
#include "nn/optimizers/sgd.h"

int main() {
    nn::LSTM lstm(3, 5);

    nn::Tensor x({2, 4, 3}, 0.0);

    for (size_t i = 0; i < x.numel(); i++) {
        x[i] = 0.01 * static_cast<double>(i);
    }

    nn::Tensor y = lstm.forward(x);

    std::cout << "LSTM output shape: [";
    for (size_t i = 0; i < y.shape().size(); ++i) {
        std::cout << y.shape()[i];
        if (i + 1 < y.shape().size()) std::cout << ", ";
    }
    std::cout << "]\n";

    nn::Tensor gy(y.shape(), 1.0);
    nn::Tensor gx = lstm.backward(gy);

    std::cout << "LSTM gradInput shape: [";
    for (size_t i = 0; i < gx.shape().size(); ++i) {
        std::cout << gx.shape()[i];
        if (i + 1 < gx.shape().size()) std::cout << ", ";
    }
    std::cout << "]\n";

    nn::SGD opt(lstm.parameters(), 0.01);
    opt.step();

    std::cout << "LSTM forward/backward/step succeeded.\n";


    return 0;
}
