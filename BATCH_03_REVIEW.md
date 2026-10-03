# Batch 03 Review — Chapters 07–09

## Chapters

- 07 — Logistic Regression
- 08 — K-Nearest Neighbors
- 09 — Naive Bayes

## Concepts Learned

### Classification

- binary target
- score/logit/probability/class
- threshold
- confusion matrix
- accuracy
- precision
- recall
- specificity
- F1
- ROC-AUC
- PR-AUC
- class imbalance
- probability calibration concept

### Logistic Regression

- sigmoid
- log odds
- Bernoulli likelihood
- binary cross-entropy
- gradient
- regularization
- multiclass/softmax concept

### KNN

- Euclidean/Manhattan/Minkowski distance
- instance-based learning
- k bias/variance
- distance weighting
- curse of dimensionality
- brute-force latency
- KDTree/BallTree concepts

### Naive Bayes

- Bayes theorem
- prior/likelihood/posterior
- conditional independence
- GaussianNB
- MultinomialNB
- BernoulliNB concept
- Laplace smoothing
- log-space computation

## Implemented From Scratch

- stable sigmoid
- binary log loss
- gradient-descent Logistic Regression
- confusion counts
- precision/recall/specificity/F1
- Minkowski distance
- KNN classifier
- Gaussian Naive Bayes
- Multinomial Naive Bayes

## scikit-learn Comparison

- LogisticRegression
- KNeighborsClassifier
- GaussianNB
- MultinomialNB
- StandardScaler
- Pipeline
- classification metrics

## Integration Project

[Classification Model Lab](integration-project-batch03/README.md)

It enforces:

```text
same split
→ same evaluation protocol
→ validation model selection
→ validation threshold selection
→ one final test
```

## Adversarial Audit

### Leakage

- threshold is selected on validation only
- final test not used for candidate selection
- scaler resides inside Pipeline
- split is stratified

### Numerical stability

- sigmoid has stable positive/negative branches
- log loss clips probabilities
- Naive Bayes uses log likelihoods
- Gaussian variance has smoothing

### Algorithm diversity

Batch compares three fundamentally different ideas:

```text
Logistic = discriminative parametric
KNN      = local instance-based
NB       = generative probabilistic
```

### Known boundaries

Not yet covered:

- tree models
- ensemble learning
- boosting
- SVM
- multiclass evaluation in depth

These belong to later chapters.

## Mastery Checklist

- [ ] derive logistic loss
- [ ] derive logistic gradient
- [ ] calculate classification metrics by hand
- [ ] explain threshold trade-offs
- [ ] implement KNN
- [ ] explain curse of dimensionality
- [ ] derive Naive Bayes posterior score
- [ ] explain smoothing/log-space
- [ ] compare discriminative/local/generative approaches
- [ ] integration project tests pass

## Exit Gate

ก่อน Chapter 10:

1. all Batch 03 tests pass
2. complete coding exercises ≥80%
3. run integration project with multiple seeds
4. explain why model selection and threshold selection belong on validation
5. compare Logistic/KNN/NB failure modes without memorized slogans
