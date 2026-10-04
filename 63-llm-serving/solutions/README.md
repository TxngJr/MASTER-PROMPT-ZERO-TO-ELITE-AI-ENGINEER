# Chapter 63 Solutions — Key Ideas

- serving includes queueing, scheduling, observability, security and rollout management in addition to model.generate().
- Transformers provides direct framework control; vLLM specializes in high-throughput GPU serving; llama.cpp is especially strong for GGUF/local hardware-flexible inference.
- bounded queues/backpressure prevent overload from becoming unbounded latency.
- p99, TTFT and token throughput answer different operational questions.
- authentication should be enforced at a real application/network boundary rather than assuming one inference-server flag protects every route.
