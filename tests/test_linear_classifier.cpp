#include "preprocess/csv_loader.h"
#include "dataset.h"
#include "models/linear/linear_classifier.h"
#include "regularization/l2_regularizer.h"
#include "regularization/no_regularizer.h"
#include "loss/perceptron_loss.h"
#include <iostream>
#include <random>

void printAccuracy(int correct, int total) {
    std::cout << "Accuracy: " << static_cast<double>(correct) / total << "\n";
}

void printTruePredScore(const std::vector<Sample>& train, const LinearClassifier& clf) {
    for (const auto& s : train) {
        std::cout << "true=" << s.label
                  << " pred=" << clf.predict(s.features)
                  << " score=" << clf.score(s.features)
                  << "\n";
    }
}

std::vector<Sample> makeLinearRuleDataset(size_t nSamples) {
    std::vector<Sample> data;
    data.reserve(nSamples);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    for (size_t i = 0; i < nSamples; i++) {
        double x1 = dist(rng);
        double x2 = dist(rng);

        double s = 2.0 * x1 - x2 + 0.5;
        int label = (s >= 0.0) ? 1 : -1;
        // if (i % 10 == 0) label = -label; // label noise

        data.push_back({{x1, x2}, label});
    }

    return data;
}

int main() {
    std::vector<Sample> simple = {
    {{-2.0, -1.0}, -1},
    {{-1.5, -1.2}, -1},
    {{-1.0, -2.0}, -1},
    {{ 1.0,  1.0},  1},
    {{ 1.5,  1.2},  1},
    {{ 2.0,  1.8},  1}
    };

    // const int numSamples = 500;
    // std::vector<Sample> train1 = makeLinearRuleDataset(numSamples);

    std::vector<Sample> train = loadDatasetCSV("tests/binary_dataset.csv");

    PerceptronLoss loss;
    L2Regularizer reg(0.001);
    // NoRegularizer reg;

    LinearClassifier clf(10, loss, reg);

    SGDOptions opts;
    opts.epochs = 10;
    opts.learningRate = 0.03;
    opts.shuffleEachEpoch = false;

    clf.fit(train);

    int correct = 0;
    for (const auto& sample : train) {
        int pred = clf.predict(sample.features);
        if (pred == sample.label) correct++;
    }

    printAccuracy(correct, train.size());

    return 0;
}
