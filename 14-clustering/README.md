# Chapter 14 — Clustering: K-Means, DBSCAN, Hierarchical, Gaussian Mixtures

## 1. Why This Matters

Clustering เป็น unsupervised learning:

```text
X
↓
ไม่มี target y
↓
หา structure / groups จากข้อมูลเอง
```

แต่ “cluster” ไม่มี definition เดียว

แต่ละ algorithm นิยาม cluster ต่างกัน:

- K-Means → compact groups around centroids
- DBSCAN → dense connected regions
- Hierarchical → nested merge/split structure
- Gaussian Mixture → probabilistic mixture components

## 2. Learning Objectives

เมื่อจบบทนี้คุณควร:

- derive K-Means objective
- implement Lloyd algorithm
- implement K-Means++ initialization
- explain inertia
- use silhouette score carefully
- implement DBSCAN core/border/noise logic
- explain eps / min_samples
- explain agglomerative linkage
- understand dendrogram
- derive Gaussian Mixture likelihood concept
- understand EM: responsibility → parameter update
- compare hard vs soft clustering
- detect when clustering output is not meaningful

## 3. Clustering Is Not Ground Truth

ถ้า algorithm สร้าง 4 clusters ไม่ได้แปลว่าโลก “มี 4 กลุ่มจริง”

ผลขึ้นกับ:

- representation
- scaling
- metric
- hyperparameters
- sampling
- algorithm assumptions

## 4. Scaling

K-Means / DBSCAN ใช้ distance จึง scale sensitive

```text
StandardScaler
↓
clustering
```

แต่ถ้า units มี semantic importance การ scale แบบอัตโนมัติอาจทำลาย meaning ได้เช่นกัน

## 5. K-Means Objective

มี K centroids:

```text
μ₁,...,μ_K
```

assignment:

```text
c_i ∈ {1,...,K}
```

objective:

```text
J = Σ_i ||x_i - μ_{c_i}||²
```

หรือ within-cluster sum of squares

scikit-learn เรียก quantity นี้ว่า inertia

## 6. Lloyd Algorithm

repeat:

1. Assignment step

```text
c_i ← argmin_k ||x_i-μ_k||²
```

2. Update step

```text
μ_k ← mean of points assigned to cluster k
```

จน labels/centroids stabilize หรือถึง max iterations

แต่ละ iteration ไม่เพิ่ม objective แต่ final solutionอาจเป็น local optimum

## 7. Why Initialization Matters

random centroids ต่างกัน → final clusters ต่างกัน

จึง:
- run multiple initializations
- keep best inertia

## 8. K-Means++

แทนสุ่ม centroid uniform:

1. เลือก center แรก
2. คำนวณ squared distance ไป center ที่ใกล้ที่สุด
3. sample center ใหม่โดย probability proportional ต่อ squared distance
4. repeat

ช่วยกระจาย initial centers และมัก convergence ดีขึ้น

## 9. Choosing K

ไม่มีวิธี universal

### Elbow
plot K vs inertia

### Silhouette

สำหรับ sample i:

```text
a(i) = mean distance within own cluster
b(i) = minimum mean distance to another cluster

s(i) = (b-a)/max(a,b)
```

ใกล้ 1 → separated  
ใกล้ 0 → boundary/overlap  
ติดลบ → possibly assigned poorly

อย่าเลือก K จาก silhouette อย่างเดียวถ้า domain meaning สำคัญ

## 10. K-Means Failure Cases

- non-spherical clusters
- unequal density/variance
- outliers
- high dimension
- irrelevant features
- categorical raw data

## 11. DBSCAN

DBSCAN ใช้ density ไม่ต้องกำหนด K

parameters:

- `eps` — neighborhood radius
- `min_samples` — minimum neighborhood mass/count สำหรับ core point

## 12. Core / Border / Noise

### Core point

มีอย่างน้อย min_samples points ภายใน eps-neighborhood โดยนับตัวมันเองตาม common implementation

### Border point

ไม่เป็น core แต่ reachable จาก core cluster

### Noise

ไม่ density-reachable จาก cluster

scikit-learn ใช้ label:

```text
-1
```

สำหรับ noise

## 13. Density Reachability

DBSCAN สร้าง cluster ด้วยการ expand จาก core points ไปยัง neighbors และต่อไปยัง core neighbors

ข้อดี:
- cluster รูปร่าง arbitrary
- detect noise

ข้อเสีย:
- eps เดียวลำบากเมื่อ densities ต่างมาก
- scale/metric sensitive
- high-dimensional distances มีปัญหา

## 14. DBSCAN Memory Note

scikit-learn implementation สามารถ bulk-compute neighborhoods และใช้ memory มากขึ้นตาม neighborhood density

บน dataset ใหญ่ต้อง monitor memory และอาจใช้ sparse radius graph/OPTICS ตาม use case

## 15. Hierarchical Clustering

Agglomerative:

```text
ทุก point เริ่มเป็น cluster
↓
merge closest pair
↓
repeat
↓
one hierarchy
```

จากนั้นตัด dendrogram ที่ระดับหนึ่งเพื่อได้จำนวน clusters

## 16. Linkage

ระยะระหว่าง clusters นิยามได้หลายแบบ

### Single

minimum pairwise distance

ข้อดี: detect elongated structure  
ข้อเสีย: chaining

### Complete

maximum pairwise distance

compact clusters มากขึ้น

### Average

average pairwise distance

### Ward

merge ที่เพิ่ม within-cluster variance น้อยที่สุด

มักใช้กับ Euclidean geometry

## 17. Dendrogram

tree ของการ merge:

```text
samples
  /
 cluster
     /
 bigger cluster
    ...
```

ช่วย inspect multiscale structure

## 18. Gaussian Mixture Model

assume data generated from mixture:

```text
p(x)=Σ_k π_k N(x|μ_k,Σ_k)
```

- `π_k` mixture weight
- `μ_k` mean
- `Σ_k` covariance

ต่างจาก K-Means เพราะ assignment เป็น probability

## 19. Responsibilities

posterior membership:

```text
r_ik = P(z_i=k | x_i)
```

แต่ละ row:

```text
Σ_k r_ik = 1
```

## 20. EM Algorithm

### E-step

คำนวณ responsibilities จาก current parameters

### M-step

update:

```text
N_k = Σ_i r_ik

π_k = N_k/n
μ_k = (1/N_k)Σ_i r_ik x_i
Σ_k = weighted covariance
```

repeat จน log-likelihood improvement เล็ก

## 21. K-Means vs GMM

K-Means:
- hard assignment
- spherical-ish distance assumption
- centroid only

GMM:
- soft assignment
- covariance shape
- probabilistic density

K-Means สามารถมองเป็น limiting/simplified case ของ mixture geometryบางแบบ แต่ไม่ควรสรุปว่าทั้งสองเหมือนกัน

## 22. Number of GMM Components

ใช้ validation likelihood / information criteria:

### AIC
trade-off fit vs parameter count

### BIC
penalty complexity แรงขึ้นตาม sample size

ยังต้อง inspect domain validity

## 23. From Scratch

[src/clustering.py](src/clustering.py)

มี:

- KMeansFromScratch
- k-means++ initialization
- inertia
- DBSCANFromScratch

GMM EM จะทำเป็น guided extension เพื่อให้ผู้เรียน derive responsibility/covariance updates เองก่อนใช้ sklearn

## 24. scikit-learn

- `KMeans(init="k-means++", n_init="auto")`
- `DBSCAN`
- `AgglomerativeClustering`
- `GaussianMixture`

## 25. Cluster Evaluation

ถ้ามี true labels ใช้ external metricsได้ แต่ clustering ไม่ควรแอบ tune เพื่อ match labels แล้วเรียก unsupervised evaluation

internal metrics:
- silhouette
- Calinski-Harabasz
- Davies-Bouldin

แต่ทุก metricมี assumptions

## 26. Common Mistakes

1. clustering raw unscaled distance features
2. K-Means กับ arbitrary-shaped clusters
3. silhouette สูง = true semantic groups
4. DBSCAN eps copy จาก tutorial
5. tune clusteringบน test labels
6. ignore noise label -1
7. hierarchical linkage ไม่เข้าใจ
8. GMM covariance singular
9. assume component = real-world class
10. cluster IDs 0/1/2 มี ordinal meaning

## 27. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] K-Means objective
- [ ] Lloyd steps
- [ ] K-Means++
- [ ] inertia
- [ ] silhouette
- [ ] DBSCAN core/border/noise
- [ ] eps/min_samples
- [ ] linkages
- [ ] dendrogram
- [ ] GMM/EM
- [ ] hard vs soft clustering

## 29. What's Next

Chapter 15 จะลด dimension เพื่อ compression, visualization และ structure discovery: PCA, SVD, t-SNE, UMAP
