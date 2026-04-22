/* File: linear_classifier.h */
#pragma once

#include "dataset.h"
#include <cinttypes>
#include <vector>
#include <cstddef>
#include <stdexcept>

class LinearClassifier {
private:
    std::vector<double> W;
    double w0;

public:
    explicit LinearClassifier(size_t numFeatures) : W(numFeatures, 0.0), w0(0.0) {}

    double score(const std::vector<double>& features) const {
        if (features.size() != W.size()) {
            throw std::runtime_error("Feature and weight vectors sizes mismatch");
        }

        double score = w0;
        for (size_t i = 0; i < features.size(); i++) {
            score += (W[i] * features[i]);
        }

        return score;
    }

    int predict(const std::vector<double>& features) const {
        return (score(features) >= 0.0 ? 1 : -1);
    }

    double misclassificationLoss(const std::vector<int>& y, const std::vector<double>& scores) const {
        if (y.size() != scores.size()) {
            throw std::runtime_error("True classes and predicted scores vectors size mismatch");
        }
        if (y.empty()) {
            throw std::runtime_error("Input vectors are empty");
        }

        int wrong = 0;
        for (size_t i = 0; i < y.size(); i++) {
            if (y[i] * scores[i] <= 0.0) {
                wrong++;
            }
        }

        return static_cast<double>(wrong) / static_cast<double>(y.size());
    }

    double mseLoss(const std::vector<double>& targets, const std::vector<double>& predicted) const {
        if (targets.size() != predicted.size()) {
            throw std::runtime_error("Target and prediction vectors sizes mismatch");
        }

        double sum_squared_error = 0.0;
        for (size_t i = 0; i < targets.size(); i++) {
            double error = targets[i] - predicted[i];
            sum_squared_error += error * error;
        }

        return sum_squared_error / static_cast<double>(targets.size());
    }

    void fit(const std::vector<Sample>& data, int epochs=10, double lr=1.0) {
        if (data.empty()) {
            throw std::runtime_error("Training data is empty");
        }

        for (const auto& sample : data) {
            if (sample.features.size() != W.size()) {
                throw std::runtime_error("Sample feature dimension mismatch");
            }
            if (sample.label != -1 && sample.label != 1) {
                throw std::runtime_error("Expected labels are +1 or -1");
            }
        }

        bool any_wrong = true;
        for (int epoch = 0; epoch < epochs; epoch++) {
            if (!any_wrong) break;
            for (const auto& sample : data) {
                int y = sample.label;
                double s = score(sample.features);

                if (y * s <= 0.0) { // wrong predicted class
                    for (size_t i = 0; i < W.size(); i++) {
                        W[i] += lr * y * sample.features[i];
                    }
                    w0 += lr * y;
                    any_wrong = true;
                }
            }
        }
    }

    const std::vector<double>& weights() const { return W; }
    double bias() { return w0; }
};
