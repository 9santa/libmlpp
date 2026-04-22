/* Binary Loss interface */

#pragma once

class BinaryLoss {
public:
    virtual ~BinaryLoss() = default;

    virtual double value(int y, double score) const = 0;

    // dL / d(score)
    virtual double dscore(int y, double score) const = 0;
};
