# Chapter 31 Solutions — Key Ideas

- patch count is spatial grid size, not flattened patch dimension.
- Conv2D with kernel=stride=patch_size can implement learned patch projection.
- CLS adds one token to the sequence.
- ordinary image-classification ViT uses full self-attention, not a causal mask.
- smaller patches improve spatial granularity but increase attention token count and quadratic interaction cost.
