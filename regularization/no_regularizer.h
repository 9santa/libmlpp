#pragma once
#include "regularizer.h"

class NoRegularizer : public Regularizer {
public:
    double value(const std::vector<double>& w) const override {
        return 0.0;
    }

    void addGradient(const std::vector<double>& w,
                     std::vector<double>& gradW) const override {

    }
};
