#include "models/linear/linear_classifier.h"
#include "regularization/l2_regularizer.h"
#include "loss/hinge_loss.h"
#include "trainers/sgd_trainer.h"
#include "preprocess/csv_loader.h"
#include <iostream>

void printAccuracy(int correct, int total) {
    std::cout << "Accuracy: " << static_cast<double>(correct) / total << "\n";
}

int main() {
    std::vector<Sample> train = loadDatasetCSV("tests/binary_dataset.csv");

    HingeLoss loss;
    L2Regularizer reg(0.001);
    LinearClassifier clf(10, loss, reg);

    SGDOptions opts;
    opts.epochs = 10;
    opts.learningRate = 0.03;
    opts.shuffleEachEpoch = true;

    clf.fit(train);

    int correct = 0;
    for (const auto& sample : train) {
        int pred = clf.predict(sample.features);
        if (pred == sample.label) correct++;
    }

    printAccuracy(correct, train.size());

    return 0;
}
