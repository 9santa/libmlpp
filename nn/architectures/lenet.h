#pragma once

#include "../core/sequential.h"
#include "../layers/4d_convolution.h"
#include "../layers/4d_pooling.h"
#include "../layers/flatten.h"
#include "../layers/linear.h"
#include "../layers/tanh.h"
#include "../layers/relu.h"

namespace nn {

enum class LeNetActivation {
    Tanh,
    ReLU
};

inline Sequential makeLeNetMNIST(
    LeNetActivation activation = LeNetActivation::Tanh,
    PoolMode poolMode = PoolMode::Avg
) {
    Sequential net;

    // Input: [N, 1, 28, 28]
    net.add<Conv4D>(1, 6, 5, 1, 0); // -> [N, 6, 24, 24]

    if (activation == LeNetActivation::Tanh) {
        net.add<Tanh>();
    } else {
        net.add<ReLU>();
    }

    net.add<Pool4D>(2, 2, poolMode);

    net.add<Conv4D>(6, 16, 5, 1, 0);

    if (activation == LeNetActivation::Tanh) {
        net.add<Tanh>();
    } else {
        net.add<ReLU>();
    }

    net.add<Pool4D>(2, 2, poolMode);

    net.add<Flatten>();

    net.add<Linear>(16 * 4 * 4, 120);

    if (activation == LeNetActivation::Tanh) {
        net.add<Tanh>();
    } else {
        net.add<ReLU>();
    }

    net.add<Linear>(120, 84);

    if (activation == LeNetActivation::Tanh) {
        net.add<Tanh>();
    } else {
        net.add<ReLU>();
    }

    net.add<Linear>(84, 10);

    return net;
}

} // namespace nn
