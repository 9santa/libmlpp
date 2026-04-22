#pragma once
#include "binary_loss.h"
#include <stdexcept>

class PerceptronLoss : public BinaryLoss {
public:
    double value(int y, double score) const override {
        validateLabel(y);
        double margin = y * score;
        return (margin <= 0.0) ? (-margin) : 0.0;
    }

    double dscore(int y, double score) const override {
        validateLabel(y);
        double margin = y * score;
        return (margin <= 0.0) ? static_cast<double>(-y) : 0.0;
    }
};
