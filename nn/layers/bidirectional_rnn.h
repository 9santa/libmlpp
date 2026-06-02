#pragma once

#include "nn/core/module.h"
#include "nn/core/tensor.h"
#include "nn/core/tensor_ops.h"
#include "nn/layers/vanilla_rnn.h"

#include <stdexcept>


namespace nn {

class BidirectionalRNN : public Module {
private:
    size_t inputDim_;
    size_t hiddenDim_;

    VanillaRNN forwardRNN_;
    VanillaRNN backwardRNN_;

public:
    BidirectionalRNN(size_t inputDim,
                     size_t hiddenDim,
                     unsigned int seed = 42)
        : inputDim_(inputDim),
          hiddenDim_(hiddenDim),
          forwardRNN_(inputDim, hiddenDim, seed),
          backwardRNN_(inputDim, hiddenDim, seed + 1) {
        if (inputDim_ == 0 || hiddenDim_ == 0) {
            throw std::runtime_error("BidirectionalRNN dimensions must be positive");
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 3) {
            throw std::runtime_error("BidirectionalRNN expects input [B, T, D]");
        }

        if (x.shape()[2] != inputDim_) {
            throw std::runtime_error("BidirectionalRNN input dimension mismatch");
        }

        Tensor forwardOut = forwardRNN_.forward(x);

        Tensor reversedInput = reverseTime(x);
        Tensor backwardOutReversed = backwardRNN_.forward(reversedInput);
        Tensor backwardOut = reverseTime(backwardOutReversed);

        return concatLastDim3D(forwardOut, backwardOut);
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (gradOutput.ndim() != 3) {
            throw std::runtime_error("BidirectionalRNN backward expects [B, T, 2H]");
        }

        if (gradOutput.shape()[2] != 2 * hiddenDim_) {
            throw std::runtime_error("BidirectionalRNN gradOutput hidden dimension mismatch");
        }

        auto [gradForward, gradBackward] = splitLastDim3D(gradOutput);

        Tensor gradInputForward = forwardRNN_.backward(gradForward);

        Tensor gradBackwardReversed = reverseTime(gradBackward);
        Tensor gradInputReversed = backwardRNN_.backward(gradBackwardReversed);
        Tensor gradInputBackward = reverseTime(gradInputReversed);

        return add(gradInputForward, gradInputBackward);
    }

};


} // namespace nn
