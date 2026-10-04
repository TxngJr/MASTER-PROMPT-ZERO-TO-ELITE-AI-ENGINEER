# Chapter 52 Solutions — Key Ideas

- data parallelism replicates model state and synchronizes gradients while each rank consumes a different data shard.
- all-reduce produces a common reduced tensor on all ranks; reduce-scatter keeps only a result shard per rank.
- FSDP/ZeRO reduce replicated model-state memory at the cost of extra communication.
- tensor/pipeline parallelism split model computation rather than only data.
- distributed speedup is limited by communication, synchronization and load imbalance.
