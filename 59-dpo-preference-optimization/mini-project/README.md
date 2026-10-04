# Mini Project — Offline Preference Optimizer

Create a toy prompt→response policy with chosen/rejected pairs.

Build:
- completion masks
- frozen reference log-prob cache
- standard DPO loss
- pairwise accuracy
- length audit
- held-out preference split

Compare:
- SFT-only baseline
- DPO
- educational IPO objective

Report held-out preference accuracy and task utility, not training loss alone.
