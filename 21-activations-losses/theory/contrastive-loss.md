# Contrastive Loss and Metric Learning

## Goal

Learn a representation where examples that should be similar are close and examples that should differ are separated.

For a pair of embeddings (z_i,z_j), a classical margin-based contrastive objective can use:

~~~text
D = ||z_i - z_j||_2
L = y * D^2 + (1-y) * max(0, margin-D)^2
~~~

with the convention (y=1) for a positive/similar pair and (y=0) for a negative pair. Always document the label convention because libraries/papers may reverse it.

## Why normalization matters

Cosine-based objectives often L2-normalize embeddings:

~~~text
u = z / max(||z||_2, epsilon)
cos(u,v) = u dot v
~~~

Without a consistent normalization policy, similarity scale and temperature behavior change.

## InfoNCE-style view

For a positive pair ((i,j)) among candidate negatives:

~~~text
p(j|i) =
exp(sim(i,j)/tau)
/
sum_k exp(sim(i,k)/tau)

L_i = -log p(j|i)
~~~

Temperature `tau` controls score sharpness. Small values create sharper distributions and can amplify noisy similarities.

## Implementation checklist

- define positive/negative construction before training;
- prevent duplicate/leaked evaluation examples from becoming easy positives;
- normalize embeddings if the objective assumes cosine similarity;
- use stable log-softmax rather than exponentiating large logits manually;
- report batch/candidate construction because in-batch negatives change the objective;
- evaluate representation quality with retrieval/classification plus hard-negative/error slices.

## Debugging lab

1. Create two obvious positive clusters and two negatives.
2. Verify positive distances decrease after an update.
3. Swap the pair labels intentionally and confirm the invariant fails.
4. Duplicate the same example across train/test and observe the optimistic metric.
