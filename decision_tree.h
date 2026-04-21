/* File: decision_tree.h */
#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
#include <memory>


struct Sample {
    std::vector<double> features;
    int label;
};

class DecisionTree {
private:
    struct Node {
        bool isLeaf = true;
        std::vector<double> probs; // size K; K - total number of classes
        int predictedClass = -1; // argmax of probs

        int splitFeatureIdx = -1; // column index of a feature
        double splitValue = 0.0;

        std::unique_ptr<Node> left = nullptr;
        std::unique_ptr<Node> right = nullptr;
    };

    struct SplitInfo {
        int featureIdx = -1;
        double threshold = 0.0;
        double score = 0.0;
        bool valid = false;
        std::vector<Sample> left;
        std::vector<Sample> right;
    };


    std::unique_ptr<Node> root = nullptr;
    int maxDepth;
    int numClasses;

    std::vector<int> classCounts(const std::vector<Sample>& data) const;
    std::vector<double> classProbs(const std::vector<Sample>& data) const;
    int argmaxClass(const std::vector<double>& probs) const;

    double giniImpurity(const std::vector<Sample>& data) const;
    double entropyImpurity(const std::vector<Sample>& data) const;

    std::unique_ptr<Node> buildTree(const std::vector<Sample>& data, int depth);
    SplitInfo findBestSplit(const std::vector<Sample>& data) const;

    const Node* traverse(const Node* node, const std::vector<double>& features) const;

public:
    explicit DecisionTree(int _maxDepth = 3, int _numClasses = 2) : maxDepth(_maxDepth), numClasses(_numClasses) {}

    void fit(const std::vector<Sample>& data);
    int predict(const std::vector<double>& features) const;
    std::vector<double> predictProba(const std::vector<double>& features) const;
};


inline std::vector<int> DecisionTree::classCounts(const std::vector<Sample>& data) const {
    std::vector<int> counts(numClasses, 0);

    for (const auto& sample : data) {
        if (sample.label < 0 || sample.label >= numClasses) {
            throw std::runtime_error("Sample label out of range.");
        }
        counts[sample.label]++;
    }

    return counts;
}

inline std::vector<double> DecisionTree::classProbs(const std::vector<Sample>& data) const {
    std::vector<double> probs(numClasses, 0.0);

    if (data.empty()) return probs;

    std::vector<int> counts = classCounts(data);
    const double N = static_cast<double>(data.size());

    for (int k = 0; k < numClasses; k++) {
        probs[k] = counts[k] / N;
    }

    return probs;
}


inline int DecisionTree::argmaxClass(const std::vector<double>& probs) const {
    auto dist = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())); // returns std::ptrdiff_t
    return static_cast<int>(dist);
}

inline double DecisionTree::giniImpurity(const std::vector<Sample>& data) const {
    if (data.empty()) return 0.0;

    std::vector<double> probs = classProbs(data);

    double gini = 0.0;
    for (auto p : probs) {
        gini += p * (1.0 - p);
    }

    return gini;
}

inline double DecisionTree::entropyImpurity(const std::vector<Sample>& data) const {
    if (data.empty()) return 0.0;

    std::vector<double> probs = classProbs(data);

    double entropy = 0.0;
    for (auto p : probs) {
        if (p > 0.0)
            entropy -= p * std::log(p);
    }

    return entropy;
}

inline std::unique_ptr<DecisionTree::Node> DecisionTree::buildTree(const std::vector<Sample>& data, int depth) {
    auto node = std::make_unique<Node>();

    node->isLeaf = true;
    node->probs = classProbs(data);
    node->predictedClass = argmaxClass(node->probs);

    if (depth >= maxDepth || giniImpurity(data) == 0.0) {
        return node;
    }

    SplitInfo split = findBestSplit(data);
    if (!split.valid) {
        return node;
    }

    if (split.left.empty() || split.right.empty()) {
        return node;
    }

    node->isLeaf = false;
    node->splitFeatureIdx = split.featureIdx;
    node->splitValue = split.threshold;
    node->left = buildTree(split.left, depth + 1);
    node->right = buildTree(split.right, depth + 1);

    return node;
}

inline std::pair<std::vector<Sample>, std::vector<Sample>> splitSamples(const std::vector<Sample>& data, int splitFeatureIdx, double splitValue) {
    std::vector<Sample> left;
    std::vector<Sample> right;
    for (const auto& sample : data) {
        if (sample.features[splitFeatureIdx] <= splitValue) left.push_back(sample);
        else right.push_back(sample);
    }

    return {left, right};
}

inline DecisionTree::SplitInfo DecisionTree::findBestSplit(const std::vector<Sample>& data) const {

    SplitInfo best;
    best.score = std::numeric_limits<double>::infinity();

    if (data.empty()) {
        return best;
    }

    const int numFeatures = static_cast<int>(data[0].features.size());
    for (const auto& sample : data) {
        if (static_cast<int>(sample.features.size()) != numFeatures) {
            throw std::runtime_error("Feature dimensions mismatch in training data.");
        }
    }

    for (int featIdx = 0; featIdx < numFeatures; featIdx++) {
        std::vector<double> values;
        values.reserve(data.size());

        for (const auto& sample : data) {
            values.push_back(sample.features[featIdx]);
        }

        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());

        if (values.size() < 2) continue;

        for (size_t i = 1; i < values.size(); i++) {
            double threshold = 0.5 * (values[i-1] + values[i]); // middle value

            auto [left, right] = splitSamples(data, featIdx, threshold);
            if (left.empty() || right.empty()) continue;

            double weightedGini = (
                static_cast<double>(left.size()) * giniImpurity(left) +
                static_cast<double>(right.size()) * giniImpurity(right))
                / static_cast<double>(data.size()
            );

            if (weightedGini < best.score) {
                best.score = weightedGini;
                best.featureIdx = featIdx;
                best.threshold = threshold;
                best.valid = true;
                best.left = std::move(left);
                best.right = std::move(right);
            }
        }
    }

    return best;
}

inline const DecisionTree::Node* DecisionTree::traverse(const Node* node, const std::vector<double>& features) const {
    if (node == nullptr) {
        throw std::runtime_error("Tree is empty.");
    }

    while (!node->isLeaf) {
        if (node->splitFeatureIdx < 0 || node->splitFeatureIdx >= static_cast<int>(features.size())) {
            throw std::runtime_error("Split feature index out of range.");
        }

        if (features[node->splitFeatureIdx] <= node->splitValue) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }

    return node;
}

inline std::vector<double> DecisionTree::predictProba(const std::vector<double>& features) const {
    const Node* leaf = traverse(root.get(), features);
    return leaf->probs;
}

inline int DecisionTree::predict(const std::vector<double>& features) const {
    const Node* leaf = traverse(root.get(), features);
    return leaf->predictedClass;
}

inline void DecisionTree::fit(const std::vector<Sample>& data) {
    if (data.empty()) {
        throw std::runtime_error("Train data is empty.");
    }
    root = buildTree(data, 0);
}
