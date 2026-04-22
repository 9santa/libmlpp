/* Binary Loss interface */

#pragma once

#include <stdexcept>
class BinaryLoss {
protected:
    void validateLabel(int y) const {
        if (y != 1 && y != -1) {
            throw std::runtime_error("Binary loss function expects labels -1 or +1");
        }
    }
public:
    virtual ~BinaryLoss() = default;

    virtual double value(int y, double score) const = 0;

    // dL / d(score)
    virtual double dscore(int y, double score) const = 0;
};
