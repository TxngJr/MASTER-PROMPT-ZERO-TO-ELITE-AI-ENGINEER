# Chapter 71 Solutions — Key Ideas

- sparse MoE stores many experts but activates only top-k experts per token.
- expert capacity and balancing are necessary because average load does not guarantee per-expert load.
- load-balancing auxiliary objectives regularize routing; they are not substitutes for the main task loss.
- expert parallelism turns routing into a communication problem as well as a compute problem.
- always report total parameters, active parameters, load distribution and overflow/drop behavior separately.
