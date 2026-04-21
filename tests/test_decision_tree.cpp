#include <iostream>
#include "../decision_tree.h"

int main() {
    std::vector<Sample> train = {
        {{0.10, 1.0}, 0},
        {{0.20, 1.2}, 0},
        {{0.30, 0.8}, 0},
        {{1.00, 2.0}, 1},
        {{1.20, 1.8}, 1},
        {{1.40, 2.2}, 1}
    };

    DecisionTree tree(2, 2);
    tree.fit(train);

    for (const auto& s : train) {
        std::vector<double> proba = tree.predictProba(s.features);
        int pred = tree.predict(s.features);

        std::cout << "x = [";
        for (size_t i = 0; i < s.features.size(); i++) {
            std::cout << s.features[i];
            if (i + 1 < s.features.size()) std::cout << ", ";
        }
        std::cout << "]"
                  << " true=" << s.label
                  << " pred=" << pred
                  << " proba=[";

        for (size_t k = 0; k < proba.size(); k++) {
            std::cout << proba[k];
            if (k + 1 < proba.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    std::vector<double> test1 = {0.15, 1.1};
    std::vector<double> test2 = {1.3, 2.1};

    auto p1 = tree.predictProba(test1);
    auto p2 = tree.predictProba(test2);

    std::cout << "\nExtra tests:\n";
    std::cout << "test1 pred = " << tree.predict(test1)
              << " proba = [" << p1[0] << ", " << p1[1] << "]\n";
    std::cout << "test2 pred = " << tree.predict(test2)
              << " proba = [" << p2[0] << ", " << p2[1] << "]\n";

    return 0;
}
