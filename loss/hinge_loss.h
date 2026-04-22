#pragma once
#include "binary_loss.h"
#include <algorithm>
#include <stdexcept>

class HingeLoss : public BinaryLoss {
public:
    double value(int y, double score) const override {
        validateLabel(y);
        double margin = y * score;
        double hinge = std::max(0.0, 1 - margin);
        return hinge;
    }


    double dscore(int y, double score) const override {
        validateLabel(y);
        double margin = y * score;
        return (margin < 1.0) ? static_cast<double>(-y) : 0.0;
    }
};
