# Mini Project — Loss Stability Lab

Generate logits from:
- -1000 to 1000
- moderate random values
- nearly tied multiclass rows

Compare:
1. naive sigmoid+BCE
2. stable BCE-with-logits
3. naive softmax+log
4. stable log-sum-exp cross entropy

Record:
- finite/non-finite outputs
- gradient magnitudes
- finite-difference error

Also compare MSE/MAE/Huber under increasing regression outliers.
