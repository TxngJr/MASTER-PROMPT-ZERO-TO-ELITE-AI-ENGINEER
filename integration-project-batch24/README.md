# Batch 24 Integration Project — Compression, Sparse MoE & Memory Lab

Batch 24 connects three efficiency/scaling axes:

~~~text
compress model weights/knowledge
↓
route tokens through sparse experts
↓
control long-context compute + memory
~~~

## Part A — Compression

A synthetic weight matrix is magnitude-pruned to 50% sparsity.

Teacher/student logits are compared with the same temperature-scaled KL definition used in Chapter 70.

Report:
- original/pruned nonzeros
- achieved sparsity
- masked values
- distillation loss

## Part B — Sparse MoE

A deterministic top-2 router reports:
- expert loads
- expert capacity
- dropped/overflow assignment fraction
- Switch-style load-balancing loss
- active-vs-total parameter fraction

## Part C — Long Context / Memory

Compare:
- dense causal attention pairs
- fixed sliding-window attention pairs
- KV-cache memory
- bounded sliding cache
- context packing under a token budget
- external-memory Recall@K

## Full Gate

The full run passes only when compression is numerically valid, MoE routing stays inside the overflow policy, and long-context/memory checks pass.

## Run

~~~bash
python integration-project-batch24/src/compression_moe_memory_lab.py --mode compression
python integration-project-batch24/src/compression_moe_memory_lab.py --mode moe
python integration-project-batch24/src/compression_moe_memory_lab.py --mode context
python integration-project-batch24/src/compression_moe_memory_lab.py --mode full
~~~

## PyTorch Framework Smoke

Batch 24 also compares PyTorch unstructured pruning and the PyTorch KL distillation objective against the chapter's framework-independent definitions.

PyTorch remains outside the core dependency chain and is installed separately in the framework CI.

## Required Extensions

1. distill a teacher into a smaller trainable student
2. prune/fine-tune the student iteratively
3. benchmark dense vs sparse runtime
4. train a tiny MoE router
5. plot expert load over training
6. simulate expert-parallel communication
7. inspect a real model RoPE configuration
8. evaluate position-dependent retrieval
9. compare append-all history vs retrieval memory
10. report a final quality-memory-latency Pareto frontier