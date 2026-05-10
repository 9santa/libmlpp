#pragma once
#include "parameter.h"

namespace nn {

class Module {
protected:
    bool training_ = true;

public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& gradOutput) = 0;

    virtual std::vector<Parameter*> parameters() {
        return {};
    }

    virtual void zeroGrad() {
        for (auto* p : parameters()) {
            p->zeroGrad();
        }
    }

    virtual void train() {
        training_ = true;
    }

    virtual void eval() {
        training_ = false;
    }

    bool isTraining() const {
        return training_;
    }
};

} // namespace nn
