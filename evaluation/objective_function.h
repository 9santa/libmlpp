#pragma once
#include "dataset.h"
#include "loss/binary_loss.h"
#include "regularization/regularizer.h"
#include "models/linear/linear_model.h"
#include <stdexcept>
#include <vector>

inline double averageLoss(const LinearModel& model,
                          const std::vector<Sample>& data,
                          const BinaryLoss& loss) {
    if (data.empty()) {
        throw std::runtime_error("Dataset is empty");
    }

    double sum_loss = 0.0;
    for (const auto& sample : data) {
        double score = model.score(sample.features);
        sum_loss += loss.value(sample.label, score);
    }

    return sum_loss / static_cast<double>(data.size());
}

inline double objectiveFunction(const LinearModel& model,
                                const std::vector<Sample>& data,
                                const BinaryLoss& loss,
                                const Regularizer& regularizer) {
    return averageLoss(model, data, loss) + regularizer.value(model.weights());
}
