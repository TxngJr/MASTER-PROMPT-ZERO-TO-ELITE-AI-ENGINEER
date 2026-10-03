# Mini Project — Autodiff Gradient Lab

Build a script that:

1. creates random X, W, b
2. computes XW+b → ReLU → square → mean
3. calls backward
4. prints analytical gradients
5. computes numerical gradients
6. reports absolute/relative error

Requirements:
- test W
- test b
- test X
- include broadcasting
- include one branched graph

Pass condition:
relative gradient error should normally be very small for smooth test points.
