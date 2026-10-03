# Chapter 15 — Dimensionality Reduction: PCA, SVD, t-SNE, UMAP

## 1. Why This Matters

High-dimensional data มีปัญหา:

- visualization ยาก
- distance geometry เสื่อม
- noise/irrelevant dimensions
- computation/memory สูง
- multicollinearity
- model debugging ยาก

Dimensionality reduction พยายามสร้าง representation ที่เล็กลงโดยรักษา structure บางประเภท

แต่ไม่มี algorithm ที่รักษา “ทุกอย่าง”

## 2. Learning Objectives

เมื่อจบบทนี้คุณควร:

- center/scale data อย่างถูกต้อง
- derive PCA จาก covariance eigenvectors
- เข้าใจ PCA ผ่าน SVD
- implement PCA จาก scratch
- calculate explained variance ratio
- reconstruct data และวัด reconstruction error
- distinguish PCA vs TruncatedSVD
- explain whitening
- explain t-SNE similarities/KL objective
- explain perplexity/early exaggeration
- explain UMAP n_neighbors/min_dist/metric
- avoid interpreting 2D embedding distances/clustersเกินจริง
- combine reduction with downstream ML without leakage

## 3. Linear Projection

ให้:

```text
X ∈ R^(n×d)
```

ลดเป็น k dimensions:

```text
Z = XW
```

โดย:

```text
W ∈ R^(d×k)
```

PCA เลือก W เป็น directions ที่ capture variance สูงสุด ภายใต้ orthogonality constraints

## 4. Centering

PCA เริ่มจาก:

```text
X_c = X - mean_train
```

mean ต้อง fit จาก training data ถ้า PCA อยู่ใน predictive pipeline

scikit-learn PCA **center แต่ไม่ scale features อัตโนมัติ**

ดังนั้นถ้า units ต่างกันและต้องการ equalized scales:

```text
StandardScaler → PCA
```

## 5. Variance Along Direction

unit vector `v`:

```text
||v||=1
```

projection:

```text
z = X_c v
```

variance ของ projected dataเกี่ยวข้องกับ:

```text
vᵀ S v
```

โดย S = covariance matrix

## 6. First Principal Component

solve:

```text
maximize vᵀSv
subject to vᵀv=1
```

Lagrangian:

```text
L(v,λ)=vᵀSv-λ(vᵀv-1)
```

differentiate:

```text
2Sv-2λv=0
```

ดังนั้น:

```text
Sv=λv
```

principal direction คือ eigenvector ของ covariance matrix และ variance ที่ capture คือ eigenvalue

เลือก eigenvectors ตาม eigenvalues ใหญ่สุด

## 7. Covariance Matrix

สำหรับ centered data:

```text
S = 1/(n-1) X_cᵀ X_c
```

shape:

```text
(d×n)(n×d) → d×d
```

## 8. Explained Variance Ratio

ถ้า eigenvalues:

```text
λ₁ >= λ₂ >= ...
```

component k:

```text
EVR_k = λ_k / Σ_j λ_j
```

cumulative:

```text
Σ_{j=1}^k EVR_j
```

ใช้ช่วยเลือก dimension แต่ “95% variance” ไม่ guarantee 95% predictive information

## 9. PCA via SVD

Centered matrix:

```text
X_c = UΣVᵀ
```

columns/rows ของ V ให้ principal directions

singular valuesสัมพันธ์กับ eigenvalues:

```text
λ_j = σ_j²/(n-1)
```

การใช้ SVD มัก numerical-friendly กว่าการสร้าง covariance matrix แล้ว eigendecompose เอง

scikit-learn PCA ใช้ SVD-based solversตาม shape/configuration

## 10. Transform

components matrix:

```text
W = [v₁ ... v_k]
```

```text
Z = X_c W
```

## 11. Reconstruction

```text
X_hat = ZWᵀ + mean
```

เมื่อ k<d จะมี information loss

reconstruction MSE ช่วยวัดสิ่งที่ PCA ทิ้งไป

## 12. PCA Signs

eigenvector sign arbitrary:

```text
v
```

และ:

```text
-v
```

represent direction เดียวกัน

ดังนั้น test PCA ควร compare subspace/projections หรือ absolute component alignment ไม่ใช่บังคับ sign ตรงกันเสมอ

## 13. Whitening

PCA whitening rescale projected components ให้ variance ประมาณ 1

ข้อดี:
- decorrelated + normalized component variance
- useful สำหรับบาง downstream assumptions

ข้อเสีย:
- ลบ relative variance information
- noise component อาจถูก amplify

ไม่เปิดโดยอัตโนมัติ

## 14. PCA vs TruncatedSVD

PCA:
- centers data
- dense workflows common

TruncatedSVD:
- ไม่ center โดย default
- เหมาะ sparse matrices เช่น TF-IDF
- เรียก LSA ใน text contexts บ่อย

อย่า center huge sparse matrixจน dense โดยไม่ตั้งใจ

## 15. PCA Failure Modes

PCA เป็น linear

ถ้า manifold โค้ง:

```text
Swiss roll
```

linear projection อาจ overlap structure

นี่คือที่มาของ nonlinear manifold methods

## 16. t-SNE

t-SNE สร้าง probability distributions ของ pairwise similarities ใน high-dimensional และ low-dimensional spaces แล้ว minimize KL divergence

เป้าหมายหลัก:
- preserve local neighborhoods
- visualization

ไม่ใช่ general-purpose invertible compression

## 17. t-SNE Objective Intuition

high-D similarities:

```text
p_ij
```

low-D similarities ใช้ heavy-tailed Student-t:

```text
q_ij
```

objective:

```text
KL(P||Q)
=
Σ p_ij log(p_ij/q_ij)
```

ทำให้ neighbors สำคัญมาก

## 18. Perplexity

perplexity roughly controls effective neighborhood scale

ควรทดลองหลายค่า

มันต้องน้อยกว่า number of samples ตาม implementation constraints

embedding ที่ต่างกันเมื่อ perplexity เปลี่ยนไม่ใช่ “bug”

## 19. Early Exaggeration

ช่วงต้น t-SNE amplify attractive forces เพื่อช่วย separate local groups

ค่าแรงเกินอาจทำ optimization behave แปลก

## 20. t-SNE Caveats

ห้ามสรุปจาก plot 2D อย่างง่ายว่า:

- gap ใหญ่ใน plot = distance ใหญ่จริงใน high-D
- cluster size เท่ากัน = density เท่ากัน
- 5 islands = exactly 5 true classes
- axes มี semantic meaning

t-SNE objective ไม่ได้พยายาม preserve global geometry ทั้งหมด

## 21. Pre-reduce Before t-SNE

scikit-learn docs แนะนำให้ลด dimension ก่อน เช่น PCA ~50 dimensions เมื่อ original features สูงมาก เพื่อ:

- suppress noise
- speed pairwise computations

## 22. UMAP

UMAP = Uniform Manifold Approximation and Projection

conceptual pipeline:

```text
local neighborhoods
↓
weighted fuzzy graph
↓
optimize low-dimensional graph representation
```

ใช้ได้ทั้ง visualization และ representation ในบาง workflows แต่ต้อง validate downstream behavior

## 23. UMAP n_neighbors

เล็ก:
- focus local structure

ใหญ่:
- more global context

official UMAP docs อธิบาย parameter นี้เป็น local-vs-global trade-off

## 24. UMAP min_dist

เล็ก:
- points pack tightly
- clustersดู compact

ใหญ่:
- embedding กระจาย

ดังนั้น “cluster ดูชัด” อาจเกิดจาก visualization hyperparameter ไม่ใช่ evidence ใหม่

## 25. UMAP metric

metric กำหนด geometry high-dimensional:

- euclidean
- cosine
- manhattan
- others

สำหรับ embeddings/text cosine อาจ meaningful กว่า Euclidean ตาม representation

## 26. Reproducibility

t-SNE/UMAP stochastic

กำหนด random_state เมื่อ experiment ต้อง reproducible

แต่ reproducible picture ยังไม่ได้ทำให้ interpretation ถูกต้อง

## 27. Leakage in Dimensionality Reduction

predictive pipeline:

```text
split
↓
fit scaler on train
↓
fit PCA on train
↓
transform train/validation/test
↓
fit model
```

ห้าม PCA fit full dataset ก่อน CV/test evaluation

## 28. From Scratch

[src/pca.py](src/pca.py)

มี:

- PCAFromScratch via SVD
- explained variance
- transform
- inverse_transform
- reconstruction error helper

## 29. scikit-learn

PCA:

```python
PCA(n_components=...)
```

t-SNE:

```python
TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    random_state=42
)
```

current scikit-learn uses `max_iter` rather than old `n_iter`

## 30. UMAP Optional

ติดตั้ง:

```bash
python -m pip install -r requirements-batch05-extras.txt
```

usage:

```python
import umap

embedder = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42,
)
Z = embedder.fit_transform(X)
```

## 31. Common Mistakes

1. PCA fit ก่อน split
2. ไม่รู้ว่า PCA centers แต่ไม่ scales
3. explained variance = predictive importance
4. component loading = causality
5. t-SNE islands = true clusters
6. compare t-SNE absolute axes across runs
7. UMAP cluster compactness = density truth
8. tune visualizationจนตรง labelsแล้วอ้าง unsupervised discovery
9. use 2D embedding as only evaluation
10. center sparse matrixจน memory explode
11. ignore random seed
12. inverse transform expected perfect after compression

## 32. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 33. Checklist

- [ ] covariance
- [ ] eigenvector derivation
- [ ] PCA via SVD
- [ ] explained variance
- [ ] reconstruction
- [ ] whitening
- [ ] PCA vs TruncatedSVD
- [ ] t-SNE KL/local structure
- [ ] perplexity
- [ ] UMAP n_neighbors/min_dist
- [ ] visualization caveats
- [ ] leakage-safe pipeline

## 34. What's Next

Batch 06 จะเข้าสู่ Anomaly Detection, Time Series และ Neural Network Foundations
