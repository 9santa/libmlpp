#pragma once
#include "metrics.h"
#include "objective_function.h"

struct EvaluationLinear {
    double averageLoss = 0.0;
    double objective = 0.0;
    double accuracy = 0.0;
};

inline EvaluationLinear evaluate(const LinearModel& model,
                                 const std::vector<Sample>& data,
                                 const BinaryLoss& loss,
                                 const Regularizer& regularizer) {
    EvaluationLinear eval;
    eval.averageLoss = averageLoss(model, data, loss);
    eval.objective = objectiveFunction(model, data, loss, regularizer);
    eval.accuracy = accuracy(model, data);
    return eval;
}
