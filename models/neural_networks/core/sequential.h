#pragma once
#include "module.h"
#include <memory>
#include <utility>

namespace nn {

class Sequential : public Module {
private:
    std::vector<std::unique_ptr<Module>> layers_;

public:
    Sequential() = default;

    template<typename LayerT, typename... Args>
    LayerT& add(Args&&... args) {
        auto layer = std::make_unique<LayerT>(std::forward<Args>(args)...);
        LayerT& ref = *layer;
        layers_.push_back(std::move(layer));
        return ref;
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = x;
        for (auto& layer : layers_) {
            out = layer->forward(out);
        }
        return out;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor grad = gradOutput;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    std::vector<Parameter*> parameters() override {
        std::vector<Parameter*> params;
        for (auto& layer : layers_) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    void zeroGrad() override {
        for (auto& layer : layers_) {
            layer->zeroGrad();
        }
    }

    void train() override {
        training_ = true;
        for (auto& layer : layers_) {
            layer->train();
        }
    }

    void eval() override {
        training_ = false;
        for (auto& layer : layers_) {
            layer->eval();
        }
    }
};

} // namespace nn
