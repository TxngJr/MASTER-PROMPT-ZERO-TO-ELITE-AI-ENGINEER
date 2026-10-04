# Chapter 59 Solutions — Key Ideas

- standard DPO compares the policy's chosen-vs-rejected log-probability margin with the reference model's margin.
- the sigmoid DPO loss is stable as softplus(-beta * margin_shift).
- completion masks must exclude prompt/padding tokens consistently across policy and reference.
- response-length distributions and stale reference-logprob caches are common hidden confounders.
- IPO, ORPO and KTO are related preference methods but do not share one identical objective.
