# Structure and Separation of concerns

### core/
Stores shared project types, such as dataset samples.


### models/
Stores classical ML model parameters and can:
- score(x)
- predict(x)
- applyGradient(...)

Does not know:
- hinge, logistic, other loss functions
- SGD


### nn/
Stores neural-network building blocks:
- Tensor
- Module
- layers
- losses
- optimizers
- reusable architectures

Concrete reusable neural-network architectures live in `nn/architectures/`.
Runnable training programs live in `examples/`.


### loss/
Knows:
- value(y, score)
- dscore(y, score)

Does not know:
- weights
- feature vector 'x'
- regularization
- epochs


### regularization/
Knows:
- penalty on weights
- addition to gradient

Does not know:
- target vector 'y'
- score
- data loop


### trainers/
Knows:
- how to run data
- how to collect gradient
- how to update the model

Does not know:
- actual math behind loss, only calls interface


### evaluation/
Knows:
- how to calculate objective
- accuracy
- training history

But doesn't train the model
