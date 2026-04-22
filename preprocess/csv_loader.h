#pragma once

#include "../dataset.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

inline std::vector<Sample> loadDatasetCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<Sample> data;
    std::string line;

    // Skip header
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;

        while (std::getline(ss, cell, ',')) {
            cells.push_back(cell);
        }

        if (cells.size() < 2) {
            throw std::runtime_error("Invalid CSV row: " + line);
        }

        std::vector<double> features;
        for (size_t i = 0; i + 1 < cells.size(); i++) {
            features.push_back(std::stod(cells[i]));
        }

        int label = std::stoi(cells.back());
        data.push_back({features, label});
    }

    return data;
}
