# Chapter 11 — Random Forest

## 1. Why This Matters

Decision Tree เดี่ยวมี variance สูง: เปลี่ยน training samples เล็กน้อย tree อาจเปลี่ยนโครงสร้างมาก

Random Forest ลดปัญหานี้ด้วยสองแหล่ง randomness:

```text
training data
   ↓ bootstrap rows
Tree 1, Tree 2, ... Tree B
   ↓ random feature subset at each split
decorrelated trees
   ↓ average probabilities / majority vote
Random Forest prediction
```

## 2. Prerequisites

- Decision Tree
- Gini / entropy
- bootstrap sampling
- bias–variance
- probability averaging
- classification metrics

## 3. Learning Objectives

เมื่อจบบทนี้คุณควร:

- อธิบาย bagging
- implement bootstrap sampling
- อธิบายเหตุผลที่ feature randomness decorrelates trees
- implement Random Forest จากศูนย์
- average class probabilities
- คำนวณ Out-of-Bag score
- tune n_estimators / max_depth / max_features
- อธิบาย OOB limitations
- เปรียบเทียบ single tree vs forest
- อธิบาย MDI importance caveats
- เข้าใจ Extra Trees concept

## 4. Bagging

Bagging = Bootstrap Aggregating

แต่ละ estimator ฝึกด้วย bootstrap sample:

```text
original indices:
0 1 2 3 4 5

bootstrap:
2 2 5 1 4 2
```

sampling **with replacement**

บาง rows ถูกเลือกหลายครั้ง บาง rows ไม่ถูกเลือกเลย

## 5. Why Bootstrap Helps

แต่ละ tree เห็นข้อมูลต่างกัน จึงสร้าง model errors ต่างกัน

ถ้า errors ไม่ perfectly correlated การ average ลด variance

intuition:

```text
Var(mean of B independent-ish models)
≈ variance / B
```

แต่ trees ไม่ independent จริง ดังนั้น key คือทำให้ correlation ลดลงด้วย random feature selection

## 6. Random Feature Subsets

ถ้าทุก tree มี feature เด่นมาก feature เดิมอาจถูกเลือก root ทุกต้น ทำให้ trees คล้ายกัน

Random Forest จึงเลือก subset ของ features ที่แต่ละ split

classification default heuristic ที่พบบ่อย:

```text
max_features ≈ sqrt(d)
```

แต่ต้อง tune ตามงาน

## 7. Aggregation

classification:

```text
P(class=k|x)
=
(1/B) Σ_b P_b(class=k|x)
```

จากนั้น:

```text
ŷ = argmax_k average probability
```

probability averaging มีข้อมูลมากกว่าการ vote class label อย่างเดียว

## 8. Out-of-Bag Samples

bootstrap sample ของ n rows มักไม่เลือกบาง rows

row ที่ไม่ถูกเลือกสำหรับ tree นั้นคือ **out-of-bag (OOB)**

ใช้ trees ที่ row นั้นเป็น OOB เพื่อทำนาย row นั้น แล้ว aggregate

ได้ OOB estimate โดยไม่ต้องมี validation splitเพิ่มเติมสำหรับบาง use cases

## 9. Why About 36.8% OOB?

probability ที่ sample หนึ่งไม่ถูกเลือกใน n draws:

```text
(1 - 1/n)^n
```

เมื่อ n ใหญ่:

```text
→ e^-1 ≈ 0.368
```

ดังนั้นแต่ละ tree โดยเฉลี่ยมี OOB rows ราว 36.8%

## 10. OOB Caveats

OOB ไม่ใช่เวทมนตร์:

- time/grouped data อาจยังผิด protocol
- preprocessing ที่ fit ก่อน forest อาจ leak
- hyperparameter tuning บน OOB ซ้ำมาก ๆ ทำให้ optimistic
- small dataset → OOB estimate noisy

## 11. Bias–Variance

เพิ่ม trees:

- variance ของ ensemble มักลด
- bias ไม่ได้ลดโดยตรงมากนัก
- prediction cost เพิ่ม
- memory เพิ่ม

เพิ่ม depth:
- individual tree bias ลด
- variance เพิ่ม
- forest ช่วยเฉลี่ย variance

## 12. Hyperparameters

### n_estimators

จำนวน trees

มากขึ้น:
- stable ขึ้น
- ช้าขึ้น
- diminishing returns

### max_depth / min_samples_leaf

คุม individual tree complexity

### max_features

คุม correlation vs strength

น้อย:
- trees diverse
- แต่ individual trees อ่อนลง

มาก:
- trees stronger
- แต่ correlated ขึ้น

## 13. Feature Scaling

เหมือน tree เดี่ยว Random Forest โดยทั่วไปไม่ต้อง standardize numeric features สำหรับ split-based behavior

แต่ preprocessing อื่น เช่น imputation/encoding ยังสำคัญ

## 14. Feature Importance

MDI importance ใน forest = average impurity decrease across trees

ข้อจำกัดเดิมยังอยู่:

- high cardinality bias
- correlated features
- not causal

Permutation importance มักเป็น alternative ที่ useful กว่าเมื่อ evaluate บน held-out data

## 15. Extra Trees

Extremely Randomized Trees เพิ่ม randomness:

- randomize split thresholds มากขึ้น
- มักใช้ whole sample หรือ configurable bootstrap
- train เร็วขึ้นบางกรณี
- variance ลด แต่ bias อาจเพิ่ม

scikit-learn มี `ExtraTreesClassifier`

## 16. Parallelism

Trees ส่วนใหญ่ train independent กัน

จึง parallelize ได้ดี:

```python
RandomForestClassifier(n_jobs=-1)
```

ระวัง memory/CPU thermals บน laptop

## 17. From Scratch

[src/random_forest.py](src/random_forest.py)

implement:

- bootstrap indices
- randomized tree
- feature subset per node
- probability averaging
- OOB accuracy

## 18. Hardware Perspective

บน Acer A715-43G:

Random Forest ใช้ CPU เป็นหลักใน scikit-learn

แนะนำเริ่ม:

```text
n_estimators: 50–200
max_depth: constrained ก่อน
n_jobs: 2–4 ตอนทดลอง
```

ค่อยเพิ่มหลังวัด RAM/temperature/runtime

อย่าใช้ `n_jobs=-1` โดยอัตโนมัติถ้า laptop ร้อนหรือทำงานอื่นพร้อมกัน

## 19. Failure Cases

- very high-dimensional sparse data ที่ linear model เหมาะกว่า
- extrapolation regression
- massive forest latency
- class imbalance
- leakage
- grouped/time splits ผิด
- categorical encoding ไม่เหมาะ
- distribution shift

## 20. Debugging

ตรวจ:

- bootstrap indices มี duplicates จริงไหม?
- feature subset เปลี่ยนทุก node ไหม?
- random seed reproducible ไหม?
- class order ของทุก tree เหมือนกันไหม?
- OOB row มีอย่างน้อยหนึ่ง tree ทำนายไหม?
- probability rows sum to 1 ไหม?

## 21. Common Mistakes

1. sampling without replacement แล้วเรียก bootstrap
2. random features แค่ครั้งเดียวต่อ tree
3. test set ใช้ tune forest
4. OOB กับ time series
5. n_estimators น้อยเกินจน score แกว่ง
6. huge forest โดยไม่วัด latency
7. feature importance = causality
8. inconsistent class ordering
9. preprocess full data ก่อน OOB/CV
10. assume forest never overfits

## 22. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 23. Checklist

- [ ] bootstrap
- [ ] bagging
- [ ] random feature subset
- [ ] correlation intuition
- [ ] probability aggregation
- [ ] OOB
- [ ] bias/variance
- [ ] Extra Trees concept
- [ ] CPU/memory trade-offs
- [ ] feature importance limits

## 24. What's Next

Chapter 12 จะเปลี่ยนจาก **parallel independent trees** ไปเป็น **sequential trees ที่แก้ error ของ ensemble ก่อนหน้า: Boosting**
