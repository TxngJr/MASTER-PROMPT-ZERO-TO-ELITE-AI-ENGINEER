# Contrastive Loss

Contrastive learning brings semantically related representations closer and pushes unrelated representations apart.

## Pairwise margin form

For embeddings z1 and z2, distance d = ||z1-z2||2, label y=1 for a matching pair and y=0 otherwise:

L = y d^2 + (1-y) max(0, m-d)^2

The positive term contracts matching pairs. The negative term penalizes non-matching pairs only when they fall inside margin m.

## Cosine / temperature form

Modern representation learning often normalizes embeddings and uses scaled cosine similarity:

s(i,j) = cosine(zi,zj) / tau

Lower temperature tau sharpens the softmax distribution. Very small tau can make optimization unstable or overly peaky.

## Practical checks
- normalize embeddings when the chosen objective assumes it
- inspect positive and negative similarity distributions
- prevent accidental duplicate/related examples from being labeled negative
- evaluate downstream retrieval/classification, not training loss alone
- vary temperature or margin as a controlled experiment

## Tiny exercise
Create four 2D embeddings with two matching pairs. Compute positive/negative distances by hand, then implement the pairwise margin loss and verify the values.
