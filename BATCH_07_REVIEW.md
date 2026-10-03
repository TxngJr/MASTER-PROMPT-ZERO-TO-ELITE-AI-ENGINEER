# Batch 07 Review — Chapters 19–21

## Chapters

- 19 — Backpropagation & Reverse-Mode Autodiff
- 20 — Optimizers
- 21 — Activations & Loss Functions

## Framework Built

The repository now contains a minimal trainable deep-learning stack:

~~~text
Tensor
├── dynamic computation graph
├── reverse topological backward
├── broadcasting gradients
├── matmul gradients
└── activations

Optimizers
├── SGD
├── Momentum / Nesterov option
├── AdaGrad
├── RMSProp
├── Adam
└── AdamW

Stable Objectives
├── MSE
├── MAE
├── Huber
├── BCE with logits
└── multiclass cross entropy
~~~

## Correctness Checks

### Autodiff
- chain rule tests
- branching accumulation
- broadcast bias gradient
- matmul finite-difference check
- non-scalar backward guard

### Optimizers
- quadratic convergence
- Adam first-step bias correction
- AdamW zero-gradient decay
- zero_grad reset

### Losses
- extreme-logit stability
- BCE finite-difference gradient
- multiclass CE finite-difference gradient
- Huber robust behavior

## PyTorch Mental Mapping

Course concept → PyTorch concept:

~~~text
Tensor.requires_grad → requires_grad
Tensor.grad          → .grad
Tensor.backward      → backward()
Optimizer.zero_grad  → optimizer.zero_grad()
Optimizer.step       → optimizer.step()
~~~

PyTorch current autograd docs describe reverse automatic differentiation over a dynamic operation graph and gradient accumulation in leaves, matching the conceptual model taught here.

## Integration Project

[Tiny Deep Learning Framework](integration-project-batch07/README.md)

It trains:
- 2→16→1 MLP
- Tanh hidden activation
- BCE-with-logits
- AdamW
- two-moons nonlinear data

without PyTorch autograd.

## Known Limitations Before Chapter 22

Tiny framework intentionally lacks:
- GPU
- mini-batch DataLoader abstraction
- modules/parameter registration
- serialization
- mixed precision
- CUDA kernels
- convolution
- production numerical kernels

Chapter 22 will show how PyTorch solves these engineering problems.

## Exit Gate

Before Chapter 22:

- [ ] derive chain rule through Dense layer
- [ ] gradient-check matmul
- [ ] explain gradient accumulation
- [ ] derive Adam bias correction
- [ ] explain Adam vs AdamW
- [ ] explain logits-aware BCE/CE stability
- [ ] train tiny MLP successfully
- [ ] debug NaN or zero-gradient scenarios
