# Chapter 67 Exercises

1. Classify ten failure examples as retryable/permanent/unknown.
2. Calculate exponential-backoff delays and add jitter in a simulation.
3. Add idempotency keys to an at-least-once worker.
4. Simulate a bounded queue at rho < 1 and rho > 1.
5. Implement a circuit breaker with timeout-based half-open transition.
6. Compare modulo sharding with rendezvous hashing after one node is added.
7. Check quorum-overlap configurations.
8. Detect stragglers from task-duration samples.
9. Design actor checkpoint/recovery metadata.
10. Draw control-plane vs data-plane dependencies for a model-serving cluster.

Challenge:
- reproduce task retries/resource scheduling with optional Ray
- build a fault-injection simulation for worker crash + duplicate retry
