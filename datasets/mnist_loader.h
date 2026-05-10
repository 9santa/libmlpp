#pragma once

#include "nn/core/tensor.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {

inline uint32_t readBigEndianUInt32(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);

    if (!file) {
        throw std::runtime_error("Failed to read uint32 from IDX file");
    }

    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

inline Tensor loadMNISTImages(const std::string& path,
                              size_t maxImages = 0,
                              bool normalize = true) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MNIST image file: " + path);
    }

    const uint32_t magic = readBigEndianUInt32(file);
    const uint32_t numImages = readBigEndianUInt32(file);
    const uint32_t rows = readBigEndianUInt32(file);
    const uint32_t cols = readBigEndianUInt32(file);

    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    size_t N = static_cast<size_t>(numImages);
    if (maxImages > 0) {
        N = std::min(N, maxImages);
    }

    Tensor images({N, 1, static_cast<size_t>(rows), static_cast<size_t>(cols)}, 0.0);

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < rows; ++h) {
            for (size_t w = 0; w < cols; ++w) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);

                if (!file) {
                    throw std::runtime_error("Unexpected end of MNIST image file");
                }

                double value = static_cast<double>(pixel);
                if (normalize) {
                    value /= 255.0;
                }

                images.at(n, 0, h, w) = value;
            }
        }
    }

    return images;
}

inline Tensor loadMNISTLabels(const std::string& path,
                              size_t maxLabels = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MNIST label file: " + path);
    }

    const uint32_t magic = readBigEndianUInt32(file);
    const uint32_t numLabels = readBigEndianUInt32(file);

    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    size_t N = static_cast<size_t>(numLabels);
    if (maxLabels > 0) {
        N = std::min(N, maxLabels);
    }

    Tensor labels({N}, 0.0);

    for (size_t n = 0; n < N; ++n) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);

        if (!file) {
            throw std::runtime_error("Unexpected end of MNIST label file");
        }

        labels[n] = static_cast<double>(label);
    }

    return labels;
}

} // namespace nn
