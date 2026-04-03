/* File: decision_tree.h */

#include <algorithm>
#include <cmath>
#include <iterator>
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
        bool isLeaf;
        std::vector<double> probs; // size K, K - total number of classes
        int predictedClass = -1; // argmax of probs

        int splitFeatureIdx = -1; // column index of a feature
        double splitValue;

        std::unique_ptr<Node> left = nullptr;
        std::unique_ptr<Node> right = nullptr;
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
        if (p > 0.0) // should always be true
            entropy -= p * std::log(p);
    }

    return entropy;
}

