# Batch 05 Review — Chapters 13–15

## Chapters

- 13 — Support Vector Machines
- 14 — Clustering
- 15 — Dimensionality Reduction

## Core Geometry

- dot products
- hyperplanes
- margins
- Euclidean/Minkowski neighborhoods
- covariance
- eigenvectors
- SVD
- local vs global neighborhood structure

## Implemented From Scratch

- hinge loss
- binary linear SVM with subgradient descent
- K-Means
- K-Means++ initialization
- DBSCAN
- PCA via SVD
- transform/inverse transform
- reconstruction MSE

## Library Skills

- LinearSVC
- SVC
- KMeans
- DBSCAN
- AgglomerativeClustering
- GaussianMixture
- PCA
- TruncatedSVD
- TSNE
- optional UMAP

## Interpretation Audit

### SVM
- C is not “regularization strength” in the same direction as Ridge alpha
- scaling matters
- kernel complexity has computational cost

### Clustering
- cluster IDs have no ordinal meaning
- algorithmic clusters are not automatically real-world classes
- internal metrics encode geometry assumptions

### Dimensionality Reduction
- PCA centers but does not auto-standardize
- explained variance is not predictive importance
- t-SNE/UMAP plots do not prove cluster truth
- PCA/t-SNE/UMAP fit must respect leakage boundaries

## Integration Project

[Geometry & Representation Lab](integration-project-batch05/README.md)

It compares:

```text
Scaled SVM
vs
Scaled → PCA → SVM

and separately:

Scaled → PCA → K-Means / DBSCAN
```

## Exit Gate

ก่อน Chapter 16:

1. all Batch 05 tests pass
2. derive hinge margin intuition
3. implement/tune K-Means and DBSCAN
4. derive PCA eigenvector objective
5. explain PCA-SVD relationship
6. explain t-SNE/UMAP visualization caveats
7. run integration project on multiple seeds
