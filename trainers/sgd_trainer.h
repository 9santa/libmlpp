#pragma once

#include "dataset.h"
#include "models/linear_model.h"
#include "loss/binary_loss.h"
#include "regularization/regularizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

struct SGDOptions {
    int epochs = 20;
    double learningRate = 0.03;
    bool shuffleEachEpoch = true;
    unsigned int seed = 42;
};

class SGDTrainer {
public:
    void fit(LinearModel& model,
             const std::vector<Sample>& data,
             const BinaryLoss& loss,
             const Regularizer& regularizer,
             const SGDOptions& options = {}) const {
        if (data.empty()) {
            throw std::runtime_error("Training data is empty");
        }

        for (const auto& sample : data) {
            if (sample.features.size() != model.numFeatures()) {
                throw std::runtime_error("Sample feature dimension mismatch");
            }
            if (sample.label != 1 && sample.label != -1) {
                throw std::runtime_error("Expected labels: -1 or +1");
            }
        }

        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(options.seed);

        for (int epoch = 0; epoch < options.epochs; epoch++) {
            if (options.shuffleEachEpoch) {
                std::shuffle(indices.begin(), indices.end(), rng);
            }

            for (size_t idx : indices) {
                const Sample& sample = data[idx];

                double s = model.score(sample.features);
                double g = loss.dscore(sample.label, s);

                std::vector<double> gradW(model.numFeatures(), 0.0);
                for (size_t j = 0; j < gradW.size(); j++) {
                    gradW[j] = g * sample.features[j];
                }

                regularizer.addGradient(model.weights(), gradW);

                double gradB = g; // don't regularize bias

                model.applyGradient(gradW, gradB, options.learningRate);
            }
        }
    }
};
