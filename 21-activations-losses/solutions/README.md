# Chapter 21 Solutions — Key Ideas

- stable softmax subtracts the row maximum
- BCE-with-logits avoids separately materializing unstable probability logs
- BCE logit gradient is sigmoid(logit)-target
- multiclass CE logit gradient is softmax-target_distribution
- Huber is quadratic near zero and linear for large residuals
- label smoothing changes target distribution and therefore gradients
