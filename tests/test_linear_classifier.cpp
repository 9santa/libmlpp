#include "../linear_classifier.h"
#include "../preprocess/csv_loader.h"
#include <iostream>
#include <random>

void printAccuracy(int correct, int total) {
    std::cout << "Accuracy: " << static_cast<double>(correct) / total << "\n";
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

    std::vector<Sample> train = loadDatasetCSV("binary_dataset.csv");

    LinearClassifier clf(10);
    clf.fit(train, 10, 0.5);

    int correct = 0;
    for (const auto& sample : train) {
        // std::cout << "x1, x2 = [" << sample.features[0] << ", " << sample.features[1] << "]; ";
        // std::cout << "label = " << sample.label << "\n";
        int pred = clf.predict(sample.features);
        if (pred == sample.label) correct++;
    }

    printAccuracy(correct, train.size());

    return 0;
}
