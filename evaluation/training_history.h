#pragma once
#include "evaluation.h"

struct EpochRecord {
    int epoch = 0;
    EvaluationLinear train;
    EvaluationLinear valid;
};

using TrainingHistory = std::vector<EpochRecord>;
