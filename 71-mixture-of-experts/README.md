# Chapter 71 — Mixture of Experts (MoE)

## 1. Sparse Activation

A dense Transformer activates essentially every feed-forward parameter for each token.

MoE replaces selected dense feed-forward blocks with multiple experts and a router:

~~~text
token hidden state
↓
router
↓
top-k experts
↓
expert outputs
↓
weighted combine
~~~

The model may contain many expert parameters while only a small subset is active per token.

## 2. Learning Objectives

- implement router softmax and top-k selection
- distinguish total vs active parameters
- compute expert capacity
- handle routing overflow
- understand token dropping/overflow policies
- derive load-balancing diagnostics
- understand auxiliary router losses
- explain expert collapse
- understand expert parallelism
- reason about all-to-all communication
- distinguish training and serving bottlenecks
- understand current Mixtral/DeepSpeed concepts

## 3. Router

For token representation h:

~~~text
router_logits = W_r h
router_prob = softmax(router_logits)
~~~

Then select top-k experts.

## 4. Top-K Routing

For top-2 routing:

~~~text
experts = arg top2(router_prob)
~~~

Selected routing weights are commonly renormalized before combining expert outputs.

## 5. Expert Output

For selected experts E_i:

~~~text
y = sum_i gate_i * E_i(h)
~~~

Only selected experts process that token.

## 6. Total vs Active Parameters

Suppose:

~~~text
8 experts
top-2 routing
~~~

All expert weights may still need to be stored, while each token computes only through two experts.

Therefore:

~~~text
large total parameter count
!= large active FLOPs per token
~~~

## 7. Mixtral Example

Current Transformers Mixtral documentation describes eight experts per MoE MLP with top-2 token routing and exposes router logits / auxiliary routing loss. (see references.md)

This is one architecture, not a universal MoE rule.

## 8. Expert Load

Count assignments:

~~~text
load_i = number of token-expert assignments to expert i
~~~

Badly imbalanced routing wastes experts and can overload a few devices.

## 9. Expert Capacity

Average assignments per expert:

~~~text
tokens * top_k / num_experts
~~~

One simple capacity rule:

~~~text
capacity = ceil(
capacity_factor
* tokens * top_k / num_experts
)
~~~

## 10. Overflow

If an expert receives more assignments than capacity, the system needs a policy:

- drop overflow assignments
- route to another expert
- increase capacity
- use expert-choice-style alternatives

Token dropping can damage quality if frequent.

## 11. Load-Balancing Loss

One Switch-style educational form:

~~~text
L_aux = E * sum_i(f_i * p_i)
~~~

where:
- f_i = fraction of top-1 assignments to expert i
- p_i = mean router probability for expert i
- E = number of experts

A perfectly uniform case gives approximately 1 under this formulation.

Current Transformers Mixtral exposes an auxiliary load-balancing loss coefficient and router logits; newer docs also mention router z-loss/load-balancing terms. (see references.md)

## 12. Expert Collapse

If router sends almost everything to a few experts:

- capacity overflows
- unused experts learn slowly
- hardware imbalance grows

Mitigations include auxiliary losses, router noise/jitter and routing-policy changes.

## 13. Router Jitter

Small training-time router noise can encourage exploration/load balance.

Do not add arbitrary inference-time randomness without understanding model implementation.

## 14. Expert Specialization

Experts may specialize by:
- token pattern
- syntax
- domain
- latent feature combinations

But routing patterns alone are not guaranteed human-readable semantics.

## 15. Dense vs Sparse MoE

Dense mixture:
~~~text
all experts active
~~~

Sparse MoE:
~~~text
only top-k active
~~~

Sparse activation is what provides the compute-vs-parameter advantage.

## 16. Expert Parallelism

Place different experts on different devices.

Tokens must move to the devices owning their selected experts.

This creates communication that dense local MLPs do not have.

## 17. All-to-All Communication

Conceptually:

~~~text
tokens on each GPU
↓ route
all-to-all exchange
↓ experts execute
all-to-all return
↓ combine
~~~

Communication can dominate if tokens are small, network is slow or expert placement is poor.

## 18. DeepSpeed MoE

Current DeepSpeed MoE documentation supports expert parallelism and combinations with other forms of parallelism; its inference material also emphasizes communication scheduling across expert parallelism. (see references.md)

## 19. Memory

MoE can reduce compute per token relative to an equally large dense model, but all expert parameters still consume aggregate storage/memory somewhere in the system.

This is why MoE can be compute-efficient yet memory-heavy.

## 20. Serving

Serving concerns:
- expert placement
- batch composition
- per-expert load
- communication
- routing kernels
- quantization
- replica memory

Tail latency can be controlled by the slowest overloaded expert/device.

## 21. Fine-Tuning

Fine-tuning MoE models adds questions:
- update router?
- update experts?
- adapter per expert?
- preserve load balance?

Always monitor routing behavior after adaptation.

## 22. From Scratch

`src/moe.py` implements:
- stable_softmax
- top_k_router
- expert_load
- expert_capacity
- apply_capacity
- dropped_assignment_fraction
- switch_load_balance_loss
- active_parameter_fraction

## 23. Common Mistakes

1. total parameters confused with active parameters
2. top-k weights not renormalized when required
3. expert capacity ignored
4. routing imbalance not measured
5. auxiliary loss treated as task loss
6. expert memory ignored
7. expert parallel communication ignored
8. router specialization overinterpreted
9. dropped-token rate not tracked
10. dense-model latency assumptions reused for MoE

## 24. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] router
- [ ] top-k
- [ ] capacity
- [ ] overflow
- [ ] load balancing
- [ ] expert collapse
- [ ] active parameters
- [ ] expert parallelism
- [ ] communication
- [ ] serving trade-offs

## 26. What's Next

Chapter 72 studies another scaling axis: context length and memory beyond one fixed attention window.