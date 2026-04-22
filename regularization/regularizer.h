/* Regularizer interface */

#pragma once
#include <vector>

class Regularizer {
public:
    virtual ~Regularizer() = default;

    virtual double value(const std::vector<double>& w) const = 0;

    // Adds regularization part to gradient
    virtual void addGradient(const std::vector<double>& w,
                             std::vector<double>& gradW) const = 0;
};
