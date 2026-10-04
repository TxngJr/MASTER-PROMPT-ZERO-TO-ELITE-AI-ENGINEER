# Mini Project — Fault-Tolerant AI Worker Pool

Simulate a distributed inference/data worker pool with:
- bounded queue
- deterministic request IDs
- rendezvous worker placement
- transient failures
- exponential backoff
- retry budget
- circuit breaker for one dependency
- task-duration/straggler telemetry
- checkpointed committed request IDs

Inject node/worker failures and prove that logical side effects remain idempotent.
