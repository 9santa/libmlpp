#pragma once

#include "core/dataset.h"
#include "models/linear/linear_model.h"
#include <stdexcept>

inline double accuracy(const LinearModel& model,
                       const std::vector<Sample>& data) {
    if (data.empty()) {
        throw std::runtime_error("Dataset is empty");
    }

    int correct = 0;
    for (const auto& sample : data) {
        if (model.predict(sample.features) == sample.label) correct++;
    }

    return static_cast<double>(correct) / static_cast<double>(data.size());
}
