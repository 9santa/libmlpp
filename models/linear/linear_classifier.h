#pragma once

#include "models/linear/linear_model.h"
#include "dataset.h"
#include "loss/binary_loss.h"
#include "regularization/regularizer.h"
#include "trainers/sgd_trainer.h"

class LinearClassifier {
private:
    LinearModel model_;
    const BinaryLoss& loss_;
    const Regularizer& regularizer_;
    SGDTrainer trainer_;

public:
    LinearClassifier(size_t numFeatures,
                     const BinaryLoss& loss,
                     const Regularizer& regularizer)
        : model_(numFeatures), loss_(loss), regularizer_(regularizer) {}

    void fit(const std::vector<Sample>& data, const SGDOptions& options = {}) {
        trainer_.fit(model_, data, loss_, regularizer_, options);
    }

    double score(const std::vector<double>& x) const {
        return model_.score(x);
    }

    int predict(const std::vector<double>& x) const {
        return model_.predict(x);
    }

    const std::vector<double>& weights() const {
        return model_.weights();
    }

    double bias() const {
        return model_.bias();
    }
};
