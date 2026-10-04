# Chapter 67 — AI Distributed Systems

## 1. Why Distributed AI Systems Fail Differently

Once training, data processing or inference spans multiple processes/nodes, partial failure becomes normal rather than exceptional.

~~~text
clients / jobs
↓
scheduler
↓
workers / actors
↓
object / model / checkpoint storage
↓
network
↓
results
~~~

A correct design asks what happens when any one arrow stalls, duplicates, retries or disappears.

## 2. Learning Objectives

- distinguish task and actor/stateful-worker models
- reason about partial failures
- design retries without duplicate side effects
- understand exponential backoff and jitter
- implement bounded queues and backpressure
- understand circuit breakers
- distinguish at-most-once / at-least-once / exactly-once claims
- shard work deterministically
- understand rendezvous/consistent-style placement
- reason about data locality
- detect stragglers
- understand replication/quorum basics
- separate control plane and data plane
- reason about checkpoint/recovery
- understand current Ray task/actor/fault-tolerance concepts

## 3. Failure Model

Possible failures:
- worker crash
- node crash
- process OOM
- network timeout
- dependency outage
- scheduler/control-plane issue
- slow straggler
- duplicate retry

Distributed correctness begins by writing the failure model explicitly.

## 4. Task Model

A task is conceptually:

~~~text
inputs → stateless function → outputs
~~~

Tasks are easier to retry when they are deterministic and side-effect free.

## 5. Actor Model

An actor owns mutable state:

~~~text
actor state
↓
method call 1
↓
method call 2
~~~

Actor recovery requires a policy for restoring that state.

## 6. Current Ray Concepts

Ray currently schedules tasks/actors based on declared resources and scheduling policies, and can prefer data locality for tasks with large local arguments.

Ray also separates application-level failures from system-level failures and provides retry/recovery mechanisms for tasks, actors and some object-loss cases.

## 7. Retries

Retrying transient failures can improve availability, but retries amplify load and duplicate side effects unless operations are idempotent.

Classify errors:
- retryable/transient
- permanent/validation
- unknown

Do not retry every exception blindly.

## 8. Exponential Backoff

One deterministic form:

~~~text
delay_k = min(cap, base × 2^k)
~~~

Production systems often add random jitter so many clients do not retry in lockstep.

## 9. Retry Budget

Set limits by:
- attempts
- total elapsed time
- request deadline

Unlimited retries can turn an outage into a retry storm.

## 10. Idempotency

If a request may execute more than once, use an idempotency identity:

~~~text
(operation, logical_request_id)
~~~

Then repeated executions can return/reuse one logical result instead of applying side effects twice.

## 11. Delivery Semantics

### At-most-once
May lose work; avoids duplicate delivery attempts.

### At-least-once
Retries until acknowledged; duplicates are possible.

### Exactly-once
Requires carefully defined boundaries and state/transaction semantics.

Exactly-once end-to-end is not implied by a framework advertising exactly-once state updates internally.

## 12. Backpressure

If producers can generate faster than consumers process, queues grow without bound.

Backpressure can:
- pause producers
- reject work
- reduce concurrency
- spill/buffer with explicit limits

Current Ray Data uses backpressure to control buffered work/object-store pressure rather than launching tasks without bound.

## 13. Queue Utilization

For arrival rate lambda and service rate mu:

~~~text
utilization rho = lambda / mu
~~~

If sustained rho >= 1, backlog tends to grow.

For multiple identical workers:

~~~text
aggregate service rate = workers × mu
~~~

## 14. Bounded Admission

A bounded queue protects memory and tail latency.

When full:
- reject
- wait with timeout
- shed lower-priority work

Make overload behavior explicit.

## 15. Circuit Breaker

A circuit breaker prevents repeated calls to an unhealthy dependency.

Typical states:

~~~text
CLOSED → OPEN → HALF_OPEN → CLOSED
~~~

CLOSED: calls allowed.
OPEN: calls rejected quickly.
HALF_OPEN: limited probe calls test recovery.

## 16. Bulkheads

Separate resource pools for unrelated workloads.

One overloaded model or tenant should not consume every worker/thread/connection.

## 17. Sharding

Split keys/work across shards:

~~~text
shard = hash(key) mod N
~~~

Modulo hashing is simple but moves many keys when N changes.

## 18. Rendezvous Hashing

For every node, compute a deterministic score for the key and choose the highest-scoring nodes.

Benefits:
- deterministic routing
- limited movement when node set changes
- simple replica selection

## 19. Replication

Replication improves availability/read capacity but creates consistency questions.

With N replicas and read/write quorums R and W, a classic overlap condition is:

~~~text
R + W > N
~~~

Real systems have additional failure, versioning and consistency details.

## 20. Control Plane vs Data Plane

Control plane:
- scheduling
- membership
- metadata
- orchestration

Data plane:
- tensors
- objects
- model requests
- gradients

Keeping these roles conceptually separate helps diagnose bottlenecks.

## 21. Data Locality

Moving a 10 GB dataset to a worker can be more expensive than moving computation to the data.

Schedulers may consider:
- resource fit
- data locality
- load
- affinity/anti-affinity

Current Ray default task scheduling can prefer nodes holding large task arguments locally.

## 22. Stragglers

One slow worker can delay a synchronized stage.

Detect using distributions rather than one hard-coded duration.

Possible mitigations:
- speculative duplicate execution for safe tasks
- smaller work units
- rebalance shards
- remove unhealthy worker

## 23. Heartbeats and Leases

Failure detectors infer liveness from missed signals/timeouts.

Timeouts trade:
- fast failure detection
- false positives during temporary pauses/network delay

## 24. Distributed Checkpoints

A checkpoint should identify:
- model/state version
- optimizer/state version
- input progress / source offsets
- shard ownership where relevant
- code/config revision

Recovery must know which work is already committed.

## 25. Object / Intermediate Data

Large distributed jobs can fail from memory pressure even if compute is available.

Plan:
- object-store capacity
- spilling
- streaming/pipelining
- backpressure
- lifetime/reference ownership

## 26. Exactly-Once Side Effects

External side effects such as billing, emails or database writes need their own idempotency/transaction boundary.

A retried distributed task does not automatically make those effects exactly once.

## 27. Distributed Training Connection

Chapter 52 covered collective training. This chapter adds systems concerns around:
- elastic worker failure
- job orchestration
- checkpoint recovery
- data service reliability
- shared control plane

## 28. Distributed Inference Connection

Serving clusters add:
- model routing
- replica health
- load balancing
- KV/prefix locality
- request retries
- overload control

Retrying an inference request after partial streaming requires special client semantics.

## 29. From Scratch

src/distributed_systems.py implements:

- exponential_backoff
- queue_utilization
- bounded_admission
- rendezvous_nodes
- modulo_shard
- quorum_overlap
- straggler_flags
- CircuitBreakerState
- circuit_breaker_record

## 30. Common Mistakes

1. retrying non-idempotent writes blindly
2. infinite retries
3. unbounded queues
4. at-least-once called exactly-once
5. actor state assumed recoverable without checkpoint
6. sharding strategy changed without migration plan
7. data locality ignored
8. one tenant consumes all workers
9. average task time hides stragglers
10. control-plane outage confused with model failure

## 31. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 32. Checklist

- [ ] failure model
- [ ] retries/backoff
- [ ] idempotency
- [ ] backpressure
- [ ] circuit breaker
- [ ] sharding
- [ ] replication/quorum
- [ ] locality
- [ ] stragglers
- [ ] checkpoints
- [ ] control/data plane

## 33. What's Next

Chapter 68 applies defense-in-depth to AI applications: risk management, prompt/tool/retrieval boundaries and safety evaluation.