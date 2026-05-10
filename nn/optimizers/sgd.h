#pragma once

#include "../core/parameter.h"

namespace nn {

class SGD {
private:
    std::vector<Parameter*> params_;
    double lr_;

public:
    SGD(const std::vector<Parameter*>& params, double lr) : params_(params), lr_(lr) {}

    void zeroGrad() {
        for (auto* p : params_) {
            p->zeroGrad();
        }
    }

    void step() {
        for (auto* p : params_) {
            for (size_t i = 0; i < p->value.numel(); i++) {
                p->value[i] -= lr_ * p->grad[i];
            }
        }
    }

    double learningRate() const { return lr_; }

    void setLearningRate(double lr) { lr_ = lr; }
};

} // namespace nn
