# Chapter 67 Solutions — Key Ideas

- retries are safe only when the operation is side-effect free, transactional, or protected by idempotency.
- backpressure and bounded queues make overload behavior explicit instead of allowing memory/latency to grow without bound.
- rendezvous hashing gives deterministic placement with less remapping than simple modulo hashing when the node set changes.
- exactly-once claims must state the source/state/sink/side-effect boundary.
- distributed systems should expect partial failure and checkpoint enough progress/state to resume safely.
