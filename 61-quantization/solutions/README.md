# Chapter 61 Solutions — Key Ideas

- symmetric quantization uses a scale around zero; asymmetric quantization adds a zero point.
- per-channel/group scales reduce range mismatch at the cost of metadata/kernel complexity.
- GPTQ and AWQ are post-training quantization methods, while GGUF is a model file/container ecosystem used by runtimes such as llama.cpp.
- low-bit storage does not imply all compute is low-bit.
- total inference memory also includes KV cache and runtime buffers.
