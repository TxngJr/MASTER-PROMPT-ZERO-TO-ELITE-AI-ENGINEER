# Chapter 56 Solutions — Key Ideas

- LoRA represents the update as a low-rank BA matrix while leaving the pretrained W frozen.
- adapter trainable parameters scale with r(in+out), not in×out.
- alpha/r controls update scale and zero-initialized B makes the initial adapter a no-op.
- QLoRA keeps a quantized frozen base while training adapters in higher-precision compute paths.
- generic INT4/codebook quantization should not be mislabeled as NF4.
