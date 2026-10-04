# Chapter 25 Solutions — Key Ideas

- recurrence reuses W_xh and W_hh at every timestep.
- BPTT is reverse-mode autodiff on the unrolled recurrent graph.
- repeated Jacobian products explain vanishing/exploding gradients.
- batch_first changes input/output sequence layout, not hidden-state layer/direction layout.
- padding must not be mistaken for real sequence content.
- bidirectional recurrence uses future context and therefore is not causal.
