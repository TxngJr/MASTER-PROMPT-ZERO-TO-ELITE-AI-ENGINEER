# Chapter 26 — LSTM & GRU

## 1. Why Gated RNNs?

Vanilla RNN repeatedly transforms hidden state through the same recurrent dynamics.

LSTM and GRU introduce gates that learn:

- what to keep
- what to overwrite
- what to expose
- how much past state should flow forward

This creates more controlled memory paths and usually improves long-range optimization.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- derive LSTM gate equations
- explain cell state
- implement an LSTM cell from scratch
- derive GRU gate equations
- implement a GRU cell from scratch
- compare LSTM vs GRU
- understand multi-layer/bidirectional shapes
- handle variable-length sequences
- explain masking/packing
- use PyTorch nn.LSTM / nn.GRU
- debug hidden/cell-state shape mismatches

## 3. LSTM State

LSTM maintains two states:

~~~text
h_t = exposed hidden state
c_t = cell state / memory path
~~~

This is the key architectural difference from a vanilla RNN.

## 4. LSTM Gates

Using concatenation conceptually:

~~~text
[x_t, h_(t-1)]
~~~

### Forget gate

~~~text
f_t = sigmoid(
    x_t W_xf
    + h_(t-1) W_hf
    + b_f
)
~~~

Controls how much old cell state remains.

### Input gate

~~~text
i_t = sigmoid(
    x_t W_xi
    + h_(t-1) W_hi
    + b_i
)
~~~

Controls how much new candidate memory is written.

### Candidate

~~~text
g_t = tanh(
    x_t W_xg
    + h_(t-1) W_hg
    + b_g
)
~~~

### Cell update

~~~text
c_t =
f_t ⊙ c_(t-1)
+
i_t ⊙ g_t
~~~

### Output gate

~~~text
o_t = sigmoid(
    x_t W_xo
    + h_(t-1) W_ho
    + b_o
)
~~~

### Hidden output

~~~text
h_t = o_t ⊙ tanh(c_t)
~~~

## 5. Why Cell State Helps

The cell update has an additive path:

~~~text
c_t =
f_t * c_(t-1)
+
new_information
~~~

The gradient through c can travel through repeated elementwise forget-gate factors rather than repeated dense recurrent matrices alone.

This does not eliminate vanishing gradients completely, but it creates a more controllable path.

## 6. Forget Gate Interpretation

If:

~~~text
f_t ≈ 1
~~~

old memory is mostly preserved.

If:

~~~text
f_t ≈ 0
~~~

old memory is mostly erased.

The gate is learned per hidden dimension.

## 7. Forget-Bias Intuition

Historically many implementations initialize forget bias toward positive values so early training tends to preserve memory.

Modern framework defaults vary, so inspect actual initialization rather than assuming.

## 8. GRU Overview

GRU simplifies the gated design.

It usually maintains only hidden state h_t.

Core gates:
- update z_t
- reset r_t

## 9. GRU Equations

One common conceptual form:

~~~text
r_t = sigmoid(
    x_t W_xr
    + h_(t-1) W_hr
    + b_r
)

z_t = sigmoid(
    x_t W_xz
    + h_(t-1) W_hz
    + b_z
)

n_t = tanh(
    x_t W_xn
    + (r_t ⊙ h_(t-1)) W_hn
    + b_n
)

h_t =
(1-z_t) ⊙ n_t
+
z_t ⊙ h_(t-1)
~~~

PyTorch's optimized GRU applies the reset gate at a slightly different algebraic location for efficiency, so framework equations can differ subtly from original-paper notation.

## 10. Update Gate Intuition

If:

~~~text
z_t ≈ 1
~~~

keep previous hidden state.

If:

~~~text
z_t ≈ 0
~~~

replace with candidate state.

## 11. Reset Gate Intuition

The reset gate controls how much previous hidden information contributes to the candidate.

Small reset value means candidate can behave more like a fresh state.

## 12. LSTM vs GRU

LSTM:
- separate cell and hidden state
- more gates/parameters
- flexible memory control

GRU:
- one main state
- fewer parameters
- often faster/simpler

Neither is universally better.

Validate on:
- sequence length
- dataset size
- latency
- accuracy
- memory

## 13. Parameter Count

For one-direction, one-layer LSTM:

Each of 4 gates has:
- input weights: D × H
- recurrent weights: H × H
- bias terms

PyTorch stores input-hidden and hidden-hidden biases separately.

Approximate parameter count:

~~~text
4 * (
    D*H
    + H*H
    + 2H
)
~~~

GRU uses 3 gate blocks instead of 4.

## 14. PyTorch Shapes

With:

~~~python
nn.LSTM(
    input_size=D,
    hidden_size=H,
    num_layers=L,
    batch_first=True,
    bidirectional=False,
)
~~~

Input:

~~~text
(B,T,D)
~~~

Output:

~~~text
(B,T,H)
~~~

h_n:

~~~text
(L,B,H)
~~~

c_n:

~~~text
(L,B,H)
~~~

Bidirectional:

~~~text
output feature dimension = 2H
h_n/c_n first dimension = 2L
~~~

## 15. GRU Shapes

GRU returns:

~~~text
output
h_n
~~~

No separate c_n.

## 16. Final State vs Last Padded Timestep

For padded batches:

~~~text
output[:, -1]
~~~

may correspond to padding for shorter samples.

Better options:
- packed sequences
- gather true final valid timestep
- use returned final state from packed recurrence

## 17. Packed Sequences

PyTorch:

~~~python
packed = pack_padded_sequence(
    padded,
    lengths,
    batch_first=True,
    enforce_sorted=False,
)
~~~

The recurrent layer can consume packed input.

Afterward:

~~~python
pad_packed_sequence(...)
~~~

if timestep outputs are needed.

## 18. Masking

When computing sequence losses, mask padded positions.

Example concept:

~~~text
loss_per_token * valid_mask
~~~

Then normalize by number of valid positions, not padded tensor size.

## 19. Bidirectionality

Bidirectional LSTM/GRU is useful for:
- tagging
- offline sequence classification
- representation extraction

Not valid for:
- strict real-time causal forecasting
- streaming systems without future context

## 20. Dropout in PyTorch Recurrent Layers

The dropout argument applies between stacked recurrent layers when num_layers > 1.

It does not mean arbitrary dropout on every recurrent connection.

## 21. From Scratch

src/gated_rnn_numpy.py includes:
- stable sigmoid
- LSTMCellNumPy
- GRUCellNumPy
- sequence forward helpers

The goal is to inspect gate values explicitly.

## 22. Debugging Gates

Inspect:
- mean/min/max of f/i/o or r/z
- hidden-state norm
- cell-state norm
- gradient norm
- saturation near 0/1

If every gate saturates early:
- initialization may be poor
- inputs may be unscaled
- learning rate may be too high

## 23. Common Mistakes

1. swapping h and c
2. wrong gate dimension
3. wrong gate chunk order
4. using padded last timestep
5. hidden-state layer/direction shape wrong
6. forgetting bidirectional doubles features
7. assuming GRU equations identical across every implementation
8. using bidirectional recurrence for causal deployment
9. masking loss incorrectly
10. comparing RNN/LSTM/GRU with different training budgets

## 24. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] LSTM f/i/g/o
- [ ] cell state
- [ ] hidden state
- [ ] GRU r/z/n
- [ ] LSTM vs GRU
- [ ] parameter count
- [ ] multi-layer shapes
- [ ] bidirectional shapes
- [ ] packing/masking
- [ ] gate debugging

## 26. What's Next

Chapter 27 moves from sequence memory to latent-variable representation learning: Autoencoders and Variational Autoencoders.
