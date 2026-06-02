#pragma once

#include "nn/core/tensor.h"
#include "nn/layers/4d_convolution.h"
#include "nn/layers/batch_norm.h"
#include "nn/layers/relu.h"

#include <memory>
#include <stdexcept>

namespace nn {

class ResidualBlock : public Module {
private:
    Conv4D conv1_;
    BatchNorm2D bn1_;
    ReLU relu1_;
    ReLU finalRelu_;

    Conv4D conv2_;
    BatchNorm2D bn2_;

    bool use1x1Conv_;

    std::unique_ptr<Conv4D> shortcutConv_;
    std::unique_ptr<BatchNorm2D> shortcutBn_;

    static Tensor addTensors(const Tensor& a, const Tensor& b) {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("ResidualBlock tensor shape mismatch in addition (forgot 1x1 conv?)");
        }

        Tensor out = a;
        for (size_t i = 0; i < out.numel(); i++) {
            out[i] += b[i];
        }

        return out;
    }


public:
    ResidualBlock(size_t inChannles,
                  size_t outChannels,
                  size_t stride = 1,
                  unsigned int seed = 42)
        : conv1_(inChannles, outChannels, 3, stride, 1, seed),
          bn1_(outChannels),
          conv2_(outChannels, outChannels, 3, 1, 1, seed + 1),
          bn2_(outChannels),
          use1x1Conv_(stride != 1 || inChannles != outChannels) {
        if (inChannles == 0 || outChannels == 0) {
            throw std::runtime_error("ResidualBlock channles must be positive");
        }

        if (use1x1Conv_) {
            shortcutConv_ = std::make_unique<Conv4D>(
            inChannles,
                        outChannels,
            1,
                        stride,
               0,
                  seed + 2
            );

            shortcutBn_ = std::make_unique<BatchNorm2D>(outChannels);
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 4) {
            throw std::runtime_error("ResidualBlock expects input [N, C, H, W]");
        }

        Tensor main = conv1_.forward(x);
        main = bn1_.forward(main);
        main = relu1_.forward(main);

        main = conv2_.forward(main);
        main = bn2_.forward(main);

        Tensor shortcut;

        if (use1x1Conv_) {
            shortcut = shortcutConv_->forward(x);
            shortcut = shortcutBn_->forward(shortcut);
        } else {
            shortcut = x;
        }

        Tensor summed = addTensors(main, shortcut);

        return finalRelu_.forward(summed);
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradAfterRelu = finalRelu_.backward(gradOutput);

        Tensor gradMain = gradAfterRelu;
        Tensor gradShortcut = gradAfterRelu;

        gradMain = bn2_.backward(gradMain);
        gradMain = conv2_.backward(gradMain);
        gradMain = relu1_.backward(gradMain);
        gradMain = bn1_.backward(gradMain);
        gradMain = conv1_.backward(gradMain);

        if (use1x1Conv_) {
            gradShortcut = shortcutBn_->backward(gradShortcut);
            gradShortcut = shortcutConv_->backward(gradShortcut);
        }

        return addTensors(gradMain, gradShortcut);
    }

    void train() override {
        training_ = true;

        conv1_.train();
        bn1_.train();
        relu1_.train();

        conv2_.train();
        bn2_.train();
        finalRelu_.train();

        if (use1x1Conv_) {
            shortcutConv_->train();
            shortcutBn_->train();
        }
    }

    void eval() override {
        training_ = false;

        conv1_.eval();
        bn1_.eval();
        relu1_.eval();

        conv2_.eval();
        bn2_.eval();
        finalRelu_.eval();

        if (use1x1Conv_) {
            shortcutConv_->eval();
            shortcutBn_->eval();
        }
    }
};





} // namespace nn
