#pragma once
#include "regularizer.h"
#include <stdexcept>

class L2Regularizer : public Regularizer {
private:
    double lambda_;

public:
    explicit L2Regularizer(double lambda) : lambda_(lambda) {
        if (lambda_ < 0.0) {
            throw std::runtime_error("L2 lambda must be non-negative");
        }
    }

    double value(const std::vector<double>& w) const override {
        double l2norm_wi = 0.0;
        for (auto wi : w) {
            l2norm_wi += wi * wi;
        }
        // 0.5 for convenience
        return 0.5 * lambda_ * l2norm_wi;
    }

    void addGradient(const std::vector<double>& w,
                     std::vector<double>& gradW) const override {
        if (w.size() != gradW.size()) {
            throw std::runtime_error("Weight and gradient vectors size mismatch");
        }

        for (size_t i = 0; i < gradW.size(); i++) {
            gradW[i] += lambda_ * w[i];
        }
    }
};
