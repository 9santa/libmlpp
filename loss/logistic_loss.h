#pragma once

#include "binary_loss.h"
#include <cmath>

class LogisticLoss : public BinaryLoss {
private:
    double sigmoid(double score) const {
        return 1.0 / (1.0 + std::exp(-score));
    }

public:
    double value(int y, double score) const override {
        validateLabel(y);

        double margin = y * score;
        double loss = std::log(1.0 + std::exp(-margin));
        return loss;
    }

    double dscore(int y, double score) const override {
        validateLabel(y);

        return -y * sigmoid(-y * score);
    }
};
