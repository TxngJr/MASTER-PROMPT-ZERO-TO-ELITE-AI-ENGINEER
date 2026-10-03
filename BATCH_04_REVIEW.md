# Batch 04 Review — Chapters 10–12

## Chapters

- 10 — Decision Trees
- 11 — Random Forest
- 12 — Gradient Boosting / AdaBoost / XGBoost / LightGBM / CatBoost

## Concepts Learned

### Trees

- recursive partitioning
- Gini
- entropy
- information gain
- thresholds
- leaves
- pruning
- tree complexity

### Random Forest

- bootstrap
- bagging
- random feature subsets
- decorrelation
- probability aggregation
- OOB evaluation
- Extra Trees concept

### Boosting

- sequential additive learning
- AdaBoost sample weights
- exponential-loss intuition
- gradient boosting / pseudo-residuals
- shrinkage
- stochastic boosting
- early stopping
- histogram boosting

### Modern Boosting Libraries

- XGBoost second-order gradients + regularization
- XGBoost hist/approx/exact tree methods
- LightGBM histogram + leaf-wise growth
- CatBoost ordered categorical statistics / boosting
- categorical-feature handling differences

## Implemented From Scratch

- Gini impurity
- entropy
- greedy Decision Tree classifier
- randomized tree
- Random Forest classifier
- bootstrap + OOB score
- regression stump
- Gradient Boosting regressor
- AdaBoost binary classifier

## Core Framework Models

- DecisionTreeClassifier
- RandomForestClassifier
- ExtraTrees concepts
- AdaBoostClassifier
- GradientBoostingClassifier/Regressor
- HistGradientBoostingClassifier

## Optional Frameworks

- XGBoost
- LightGBM
- CatBoost

ถูกแยกเป็น optional dependencies เพื่อให้ core CI และ Fedora laptop setup ไม่หนักเกินจำเป็น

## Adversarial Audit

### Mathematics

- impurity equations define class probabilities
- weighted child impurity uses sample proportions
- AdaBoost alpha/sample update derived
- squared-loss negative gradient derived
- XGBoost first/second derivatives distinguished

### Leakage

- integration project selects model on validation only
- final test used after selection
- no target encoding across full data
- OOB limitations documented

### Systems

- forest parallelism vs boosting sequential dependency
- histogram methods explained as speed/memory optimization
- external package differences not reduced to superficial parameter names

### Interpretability

- MDI importance is explicitly not causality
- correlated/high-cardinality caveats documented

## Mastery Checklist

- [ ] calculate Gini and entropy by hand
- [ ] implement tree split search
- [ ] explain pruning
- [ ] implement bootstrap
- [ ] explain why random features decorrelate trees
- [ ] derive OOB 36.8% intuition
- [ ] derive AdaBoost alpha/update
- [ ] derive squared-loss pseudo-residual
- [ ] explain XGBoost gradient/hessian
- [ ] explain LightGBM leaf-wise risk
- [ ] explain CatBoost category leakage strategy
- [ ] run Batch 04 integration benchmark

## Exit Gate

ก่อน Chapter 13:

1. all Batch 04 tests pass
2. integration benchmark runs
3. complete coding exercises ≥80%
4. compare Tree / RF / Boosting failure modes
5. explain why XGBoost, LightGBM and CatBoost should not be compared by matching parameter names alone
