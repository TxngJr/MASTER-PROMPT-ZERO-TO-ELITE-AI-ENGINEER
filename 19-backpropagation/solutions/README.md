# Chapter 19 Solutions — Key Ideas

- chain rule multiplies local gradient by upstream gradient
- branches add gradient contributions
- matrix multiply:
  - dX = G W^T
  - dW = X^T G
- broadcasted dimensions must be summed during backward
- scalar loss backward starts from dL/dL=1
- numerical finite differences are a debugging oracle, not a training algorithm
