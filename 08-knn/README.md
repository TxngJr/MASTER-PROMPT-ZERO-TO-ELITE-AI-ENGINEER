# Chapter 08 — K-Nearest Neighbors (KNN)

## 1. Why This Matters

KNN เป็นตัวอย่างสำคัญของ **instance-based learning**: แทนที่จะเรียน parameters แบบ `w,b` มันเก็บ training examples แล้วตัดสินจากตัวอย่างที่อยู่ใกล้ query

```text
training data
   ↓ stored
new sample x
   ↓ distance to training points
k nearest neighbors
   ↓ vote / weighted vote
predicted class
```

## 2. Prerequisites

- vectors และ distance
- classification metrics
- train/validation/test
- standardization
- NumPy

## 3. Learning Objectives

เมื่อจบบทนี้คุณควร:

- อธิบาย instance-based learning
- คำนวณ Euclidean / Manhattan / Minkowski distance
- implement KNN classifier จากศูนย์
- อธิบายผลของ `k`
- อธิบาย uniform vs distance-weighted voting
- เข้าใจเหตุผลที่ scaling สำคัญ
- อธิบาย curse of dimensionality
- แยก brute-force search กับ KDTree/BallTree concept
- tune K บน validation/CV
- อธิบาย memory/latency trade-offs

## 4. Distance Metrics

### Euclidean

```text
d(x,z)=sqrt(Σ(x_j-z_j)^2)
```

### Manhattan

```text
d(x,z)=Σ|x_j-z_j|
```

### Minkowski

```text
d_p(x,z)=(Σ|x_j-z_j|^p)^(1/p)
```

- p=1 → Manhattan
- p=2 → Euclidean

## 5. Why Scaling Matters

ถ้า features:

```text
age: 0..100
income: 0..1,000,000
```

Euclidean distance จะถูก income ครอบงำ

จึงมักใช้:

```text
split
→ fit StandardScaler on train
→ transform
→ KNN
```

## 6. Choosing K

เล็กมาก:
- flexible
- sensitive ต่อ noise
- variance สูง

ใหญ่ขึ้น:
- smoother boundary
- bias สูงขึ้น

จึงเลือก `k` จาก validation/CV ไม่ใช่ test

## 7. Voting

Uniform:

```text
class = majority(neighbor labels)
```

Distance-weighted:

```text
weight_i = 1/(d_i + ε)
```

neighbor ใกล้มีอิทธิพลมากกว่า

ต้อง handle distance=0 เพื่อไม่หารศูนย์

## 8. Decision Boundaries

KNN ไม่มีสมการ boundary แบบ linear model

boundary เกิดจาก geometry ของ training points

ด้วย k=1 สามารถสร้าง boundary ซับซ้อนมาก

## 9. Complexity

Training:
- เกือบไม่มี optimization; เก็บ data

Prediction brute-force:

```text
O(n_train × d)
```

ต่อ query สำหรับ distance computation

Memory:

```text
O(n_train × d)
```

KNN จึง “train เร็ว” แต่ prediction อาจแพง

## 10. Faster Neighbor Search

แนวคิด:

- brute force
- KDTree
- BallTree
- approximate nearest neighbors

tree methods มีประโยชน์ในบาง dimension/metric แต่ high-dimensional data ทำให้ประสิทธิภาพลดลง

## 11. Curse of Dimensionality

เมื่อ dimension สูง:

- space volume โตเร็ว
- data sparse
- nearest/farthest distances อาจแตกต่างกันน้อยลง
- ต้องการ data จำนวนมากขึ้นเพื่อ cover space

ดังนั้น “ใกล้” อาจมีความหมายลดลงใน raw high-dimensional features

แนวคิดนี้จะกลับมาใน embeddings และ vector search

## 12. Ties

ถ้า vote เสมอ ต้องกำหนด deterministic policy

implementation ของ course ใช้:
1. vote count
2. ถ้าเสมอ ใช้ summed inverse-distance weight
3. ถ้ายังเสมอ เลือก label ที่ sort ได้ก่อน

policy ต้องชัดเพื่อ reproducibility

## 13. KNN for Regression

แนวคิดเดียวกัน แต่แทน vote ด้วย average/weighted average target

บทนี้ focus classification แต่ควรรู้ว่า neighbor methods ใช้ regression ได้

## 14. From Scratch

[src/knn.py](src/knn.py) มี:

- pairwise distance
- KNNClassifier
- uniform/distance weights
- deterministic tie handling

## 15. scikit-learn

`KNeighborsClassifier` รองรับ:

- `n_neighbors`
- `weights="uniform"`
- `weights="distance"`
- Minkowski metric ผ่าน `p`
- backend search algorithm แบบ auto/brute/tree

## 16. Hyperparameters

สำคัญ:

- k
- distance metric
- weighting
- preprocessing
- feature selection

อย่า tune ทุกอย่างบน test

## 17. Failure Cases

- dimension สูงมาก
- irrelevant features จำนวนมาก
- unscaled features
- huge training set
- noisy labels
- highly imbalanced local neighborhoods
- distribution shift

## 18. Debugging

ถ้า prediction แปลก:

- print nearest distances
- print neighbor indices
- inspect neighbor labels
- check feature scaling
- verify train/test preprocessing
- test query ที่ตรงกับ training point

## 19. Common Mistakes

1. ไม่ scale
2. tune k บน test
3. ใช้ k มากกว่า training samples
4. distance-weighting แล้วหาร 0
5. ไม่กำหนด tie policy
6. คิดว่า training cost ต่ำ = system เร็ว
7. ใส่ irrelevant dimensions จำนวนมาก
8. random split กับ grouped/time data
9. ใช้ Euclidean โดยไม่คิด semantics
10. ignore class imbalance

## 20. Exercises & Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 21. Checklist

- [ ] distance metrics
- [ ] scaling
- [ ] k bias/variance
- [ ] voting
- [ ] curse of dimensionality
- [ ] complexity
- [ ] implement KNN
- [ ] compare sklearn
- [ ] validation tuning

## 22. What's Next

Chapter 09 จะเปลี่ยนจาก geometry-based classifier ไปเป็น probabilistic generative classifier: **Naive Bayes**
