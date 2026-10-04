# Mini Project — Sparse MoE Router Lab

Build a tiny MoE classifier with 4–8 experts and top-1/top-2 routing.

Measure:
- router entropy
- expert loads
- capacity overflow
- dropped assignment rate
- task loss
- load-balancing auxiliary loss
- active vs total parameters

Then deliberately bias one expert and show how load balance and latency proxies degrade.
