# Mini Project — Production-Style Local LLM Service

Build a local service with:
- request validation
- bounded queue
- streaming
- cancellation
- readiness/liveness
- request IDs
- TTFT/latency/token metrics
- graceful shutdown

Then run a load test using realistic prompt/output length distributions.

Optional backends:
- Transformers
- vLLM on compatible GPU systems
- llama.cpp with GGUF

Record exact runtime/model/quantization revisions.
