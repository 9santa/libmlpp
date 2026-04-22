#pragma once
#include <stdexcept>
#include <vector>

class LinearModel {
private:
    std::vector<double> w_;
    double b_ = 0.0;

public:
    explicit LinearModel(size_t numFeatures) : w_(numFeatures, 0.0), b_(0.0) {}

    size_t numFeatures() const {
        return w_.size();
    }

    double score(const std::vector<double>& x) const {
        if (x.size() != w_.size()) {
            throw std::runtime_error("Feature dimension mismatch");
        }

        double s = b_;
        for (size_t i = 0; i < x.size(); i++) {
            s += w_[i] * x[i];
        }
        return s;
    }

    int predict(const std::vector<double>& x) const {
        return (score(x) >= 0.0) ? 1 : -1;
    }

    const std::vector<double>& weights() const {
        return w_;
    }

    double bias() const {
        return b_;
    }

    void applyGradient(const std::vector<double>& gradW,
                       double gradB,
                       double lr) {
        if (gradW.size() != w_.size()) {
            throw std::runtime_error("Gradient dimension mismatch");
        }

        for (size_t i = 0; i < w_.size(); i++) {
            w_[i] -= lr * gradW[i];
        }
        b_ -= lr * gradB;
    }
};
