# Chapter 72 Solutions — Key Ideas

- dense causal attention has quadratic token-pair growth, while fixed sliding windows approach O(NW).
- positional-extension methods such as RoPE scaling do not themselves guarantee model quality at new lengths.
- KV cache can dominate long-context inference memory; sliding/chunked layers can bound cache growth for those layers.
- active context, compressed summaries and persistent external memory are separate system layers.
- evaluate long context across position, length, distractors and realistic task behavior rather than one needle benchmark.
