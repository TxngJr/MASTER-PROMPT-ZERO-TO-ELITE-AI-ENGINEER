# Chapter 25 — Recurrent Neural Networks

## 1. Why Sequence Models?

A feed-forward network sees a fixed vector.

A sequence model receives ordered inputs:

~~~text
x1 → x2 → x3 → ... → xT
~~~

and maintains state that summarizes earlier steps.

Examples:
- text
- sensor streams
- audio frames
- event sequences
- time-series windows

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain hidden state
- derive the Elman RNN recurrence
- implement a tanh RNN forward pass from scratch
- understand sequence-to-one vs sequence-to-sequence
- explain BPTT
- explain vanishing/exploding gradients
- use gradient clipping
- reason about hidden-state shapes
- use PyTorch nn.RNN
- handle padded variable-length batches
- understand bidirectional RNN caveats
- distinguish stateful recurrence from attention

## 3. Elman RNN

At time t:

~~~text
h_t = tanh(
    x_t W_xh
    + h_(t-1) W_hh
    + b_h
)
~~~

optional output:

~~~text
y_t = h_t W_hy + b_y
~~~

The same parameters are reused at every time step.

## 4. Shapes

For batch-first input:

~~~text
X:
(B, T, D_in)

W_xh:
(D_in, H)

W_hh:
(H, H)

h_t:
(B, H)
~~~

Output sequence:

~~~text
(B, T, H)
~~~

Final hidden state for one layer/direction:

~~~text
(1, B, H)
~~~

PyTorch hidden-state layout is not changed by batch_first.

## 5. Hidden State Mental Model

The hidden state is not a perfect memory.

It is a learned compressed summary:

~~~text
past inputs
→ hidden state
→ next prediction / next state
~~~

Information can be forgotten or distorted as the sequence grows.

## 6. Sequence-to-One

Examples:
- sentiment classification
- activity classification

~~~text
x1 x2 x3 ... xT
 ↓  ↓  ↓      ↓
 recurrent state
              ↓
           final h
              ↓
          classifier
~~~

## 7. Sequence-to-Sequence

Output at every step:

~~~text
h1 → y1
h2 → y2
...
hT → yT
~~~

Used for:
- tagging
- per-timestep prediction
- autoregressive systems

## 8. Unrolling

The recurrent network can be visualized as copies across time:

~~~text
h0
 ↓
[RNN cell] ← x1
 ↓ h1
[RNN cell] ← x2
 ↓ h2
...
~~~

The copies share parameters.

## 9. Backpropagation Through Time

BPTT is ordinary reverse-mode differentiation on the unrolled graph.

Gradient to early states includes repeated Jacobian products:

~~~text
∂L/∂h_t
contains
W_hhᵀ
W_hhᵀ
...
through many steps
~~~

Repeated multiplication creates the vanishing/exploding-gradient problem.

## 10. Vanishing Gradients

If repeated effective Jacobian norms are mostly below 1:

~~~text
gradient magnitude
→ smaller
→ smaller
→ nearly zero
~~~

Long-range dependencies become difficult to learn.

## 11. Exploding Gradients

If repeated products have norms above 1:

~~~text
gradient
→ huge
→ unstable updates
→ inf / NaN
~~~

Mitigations include:
- gradient clipping
- gated architectures
- initialization choices
- normalization/optimization choices

## 12. Gradient Clipping

PyTorch:

~~~python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
~~~

Typical order:

~~~text
zero_grad
forward
loss
backward
clip
optimizer.step
~~~

Clipping is not a cure for every unstable model.

## 13. Truncated BPTT

Instead of backpropagating across an extremely long sequence, training can detach state periodically.

Concept:

~~~text
window 1 → backward
detach hidden
window 2 → backward
...
~~~

This bounds graph length and memory but limits gradient credit assignment across boundaries.

## 14. Initial Hidden State

Common default:

~~~text
h0 = zeros
~~~

Alternatives:
- learned initial state
- state carried across chunks

If carrying state, detach it between optimization chunks unless full graph retention is intentional.

## 15. Bidirectional RNN

A bidirectional RNN processes:

~~~text
forward:  1 → T
backward: T → 1
~~~

Useful when the entire sequence is available.

Not appropriate for causal streaming if the model cannot access future tokens/events at inference time.

## 16. Stacked RNN

Multiple recurrent layers:

~~~text
input
↓
RNN layer 1
↓
RNN layer 2
↓
...
~~~

PyTorch num_layers controls this.

Dropout in recurrent modules applies between recurrent layers, not as arbitrary recurrent-state dropout at every location.

## 17. Variable-Length Sequences

Padding wastes compute and can contaminate naive final-state selection.

PyTorch supports:
- pad_sequence
- pack_padded_sequence
- pad_packed_sequence
- pack_sequence

With batch_first=True, padded input can be:

~~~text
B × T × D
~~~

Lengths still describe true sequence lengths.

## 18. Masking

If you do not pack sequences, use masks so padded positions do not contribute to:
- loss
- metrics
- pooling

## 19. PyTorch nn.RNN

~~~python
rnn = nn.RNN(
    input_size=16,
    hidden_size=64,
    num_layers=2,
    batch_first=True,
    nonlinearity="tanh",
)
~~~

Outputs:

~~~text
output: hidden state for every timestep
h_n: final hidden state per layer/direction
~~~

## 20. From Scratch

src/rnn_numpy.py includes:
- stable tanh RNN cell
- sequence forward pass
- linear sequence readout
- explicit shape validation

The implementation is forward-only to isolate recurrence math.

BPTT is already conceptually supported by the Batch 07 autodiff engine; the exercises connect them.

## 21. RNN Limitations

- long-range memory
- sequential computation limits parallelism
- training instability
- hidden-state bottleneck
- causal recurrence can be slower than parallel attention for long sequences

RNNs remain useful for:
- small streaming models
- stateful low-latency systems
- compact sequence baselines

## 22. Common Mistakes

1. mixing batch/time axes
2. assuming batch_first affects h_n layout
3. using padded final timestep as real state
4. carrying hidden state without detach
5. bidirectional model in causal task
6. no gradient clipping on unstable long sequences
7. shuffling timesteps within a sequence
8. confusing batch shuffle with timestep shuffle
9. using final output incorrectly for multi-layer bidirectional RNN
10. assuming hidden state stores all history perfectly

## 23. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 24. Checklist

- [ ] recurrence equation
- [ ] hidden state
- [ ] unrolling
- [ ] BPTT
- [ ] vanishing/exploding
- [ ] clipping
- [ ] truncated BPTT
- [ ] batch/time shapes
- [ ] packed sequences
- [ ] bidirectional caveat

## 25. What's Next

Chapter 26 adds gates and a dedicated cell state to address long-range optimization problems: LSTM and GRU.
