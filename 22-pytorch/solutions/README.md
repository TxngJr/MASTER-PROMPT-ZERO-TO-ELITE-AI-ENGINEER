# Chapter 22 Solutions — Key Ideas

- autograd accumulates gradients unless reset.
- eval() changes module behavior but does not itself disable gradient tracking.
- inference_mode() is appropriate for inference-only sections.
- CrossEntropyLoss consumes logits and integer class indices.
- registered Parameters and child Modules appear in state_dict.
- device and dtype mismatches should be inspected explicitly rather than guessed.
