# Chapter 56 — LoRA, QLoRA & PEFT

## 1. Why PEFT?

Full fine-tuning stores gradients and optimizer state for every trainable weight.

Parameter-Efficient Fine-Tuning (PEFT) adapts a model while training a much smaller parameter subset.

~~~text
large pretrained model
↓ frozen mostly/all
small trainable adaptation parameters
↓
task-adapted model
~~~

## 2. Learning Objectives

You should be able to:

- derive LoRA's low-rank update
- calculate LoRA parameter counts
- explain rank, alpha and dropout
- initialize LoRA as a no-op
- choose target modules
- merge/unmerge adapters
- explain PEFT memory savings
- distinguish LoRA from full fine-tuning
- explain QLoRA correctly
- understand frozen 4-bit base weights
- explain NF4 / double-quantization concepts
- understand adapter checkpointing
- use current PEFT LoraConfig concepts
- reason about adapter composition and limitations

## 3. LoRA Core Idea

For frozen linear weight:

~~~text
W ∈ R^(out × in)
~~~

instead of updating W directly, learn:

~~~text
ΔW = B A
~~~

where:

~~~text
A ∈ R^(r × in)
B ∈ R^(out × r)
r << min(in,out)
~~~

Forward:

~~~text
y =
Wx
+
(alpha / r) B A x
~~~

## 4. Parameter Count

Full linear matrix:

~~~text
out * in
~~~

LoRA adapter:

~~~text
r * in
+
out * r

=
r(in + out)
~~~

Example:

~~~text
in = out = 4096
r = 8

full:
16,777,216 parameters

LoRA:
65,536 parameters
~~~

for that matrix.

## 5. Scaling Alpha

Common LoRA scale:

~~~text
scale =
alpha / r
~~~

Alpha controls update magnitude independently from rank.

Some modern variants use different scaling; always document the chosen formulation.

## 6. Initialization

A common initialization:
- A random
- B zero

Then initially:

~~~text
BA = 0
~~~

so the adapter begins as an identity/no-op relative to the base model.

Current PEFT defaults also initialize adapters so they initially do not change the base model's output.

## 7. LoRA Dropout

Dropout can be applied to adapter input during training.

This regularizes adapter updates.

Inference normally disables dropout.

## 8. Target Modules

Common transformer targets:
- q_proj
- k_proj
- v_proj
- o_proj
- gate/up/down MLP projections

Which modules are best depends on:
- architecture
- task
- rank
- memory budget

Current PEFT can target explicit module names or broader selections such as all linear modules for QLoRA-style training.

## 9. Query/Value-Only vs All-Linear

Original/simple recipes often adapt a small attention subset.

QLoRA-style PEFT commonly targets all linear layers.

Trade-off:
- more target layers → more capacity
- more trainable adapter parameters

Benchmark rather than assume.

## 10. Trainable Ratio

For base P and adapters A:

~~~text
trainable_ratio =
A / (P + A)
~~~

Often far below 1%.

## 11. Adapter Checkpoint

LoRA checkpoint stores:
- adapter configuration
- A matrices
- B matrices
- optional extra trainable modules

It usually does **not** contain the complete frozen base model.

Therefore deployment needs:
- compatible base model
- compatible adapter

## 12. Merge

At inference:

~~~text
W_merged =
W
+
(alpha/r)BA
~~~

Then the adapter path can be removed for a plain merged linear layer.

Benefits:
- no adapter matmul overhead

Trade-off:
- loses easy adapter switching unless original base is retained

## 13. Unmerge

If the original base weight is preserved:

~~~text
W =
W_merged
-
scale * BA
~~~

Repeated merge/unmerge should avoid accumulating numerical error unnecessarily.

## 14. Multiple Adapters

One base model can host:
- task A adapter
- task B adapter
- domain adapter

This is operationally cheaper than storing separate full-model checkpoints.

## 15. PEFT Family

Beyond LoRA:
- prefix tuning
- prompt tuning
- IA3
- adapters
- newer low-rank variants

The shared idea is reducing trainable state.

## 16. QLoRA

QLoRA combines:
- frozen quantized base model
- trainable LoRA adapters

Important:

> The base model is quantized for storage/computation efficiency, while adapter parameters remain trainable. QLoRA is not ordinary gradient descent directly updating 4-bit frozen base weights.

## 17. 4-bit Base Weights

Conceptually:

~~~text
FP weight block
↓ quantize
4-bit codes + scale metadata
↓
dequantize as needed for compute
~~~

Actual kernels may fuse dequantization with matrix multiplication.

## 18. NF4

NormalFloat4 was designed for normally distributed neural-network weights.

Conceptually:
- 16 representable levels
- levels chosen for the weight distribution rather than uniform spacing

This differs from simple uniform INT4.

## 19. Double Quantization

Quantization itself needs scale/constants.

Double quantization compresses some of those quantization constants too.

Goal:
- reduce metadata memory overhead

## 20. Compute Dtype

4-bit storage does not imply every arithmetic operation is performed in 4-bit.

Typical QLoRA systems:
- store base weights in 4-bit form
- dequantize into BF16/FP16-like compute path
- train adapters in higher precision

## 21. Optimizer State

Only trainable adapter parameters require ordinary gradient/optimizer state.

This is a major memory saving compared with full fine-tuning.

## 22. Memory Decomposition

Full fine-tuning roughly needs:
- base parameters
- gradients
- optimizer states
- activations

QLoRA:
- quantized frozen base
- adapter parameters
- adapter gradients
- adapter optimizer state
- activations/dequant buffers

Activations can still be large.

## 23. Gradient Checkpointing

QLoRA is often combined with activation checkpointing to further reduce memory.

This trades compute for activation memory.

## 24. Adapter Rank

Low rank:
- cheaper
- lower capacity

High rank:
- more trainable parameters
- more adaptation capacity

Rank is a hyperparameter, not a quality guarantee.

## 25. Current Hugging Face PEFT

Typical current concepts:

~~~python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config)
~~~

For QLoRA-style targeting, current PEFT supports:

~~~python
target_modules="all-linear"
~~~

Always inspect the actual model's module names/config.

## 26. Base Compatibility

An adapter is tied to:
- architecture
- layer names
- dimensions
- base checkpoint/version in practice

Loading it onto a mismatched base can fail or silently degrade behavior.

Store base model revision with adapter metadata.

## 27. From Scratch

src/lora_numpy.py includes:

- lora_parameter_count
- lora_delta
- lora_forward
- merge_lora
- unmerge_lora
- trainable_fraction
- nearest_codebook_quantize

The generic codebook quantizer is educational; it is **not** claimed to reproduce production NF4 kernels.

## 28. Common Mistakes

1. accidentally training the frozen base
2. wrong matrix orientation for A/B
3. forgetting alpha/r scaling
4. adapter initialized with nonzero update unintentionally
5. target module names do not exist
6. comparing trainable params but ignoring activations
7. saying QLoRA updates 4-bit base weights directly
8. calling uniform INT4 "NF4"
9. merging twice
10. adapter deployed with wrong base checkpoint

## 29. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 30. Checklist

- [ ] low-rank update
- [ ] parameter count
- [ ] alpha/r
- [ ] initialization
- [ ] target modules
- [ ] merge/unmerge
- [ ] PEFT checkpointing
- [ ] QLoRA
- [ ] 4-bit base
- [ ] NF4 concept
- [ ] double quantization
- [ ] memory accounting

## 31. What's Next

Chapter 57 applies fine-tuning/PEFT to instruction and chat datasets using supervised fine-tuning.
