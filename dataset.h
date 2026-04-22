/* File: dataset.h */
#pragma once
#include <vector>

struct Sample {
    std::vector<double> features;
    int label;
};
