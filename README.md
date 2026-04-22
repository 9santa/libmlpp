# Structure and Separation of concerns

### models/
Stores model parameters and can:
- score(x)
- predict(x)
- applyGradient(...)

Does not know:
- hinge, logistic, other loss functions
- SGD


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
