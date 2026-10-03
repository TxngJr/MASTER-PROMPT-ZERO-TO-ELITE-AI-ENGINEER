# Chapter 10 — Decision Trees

## 1. Why This Matters

Decision Tree เปลี่ยนวิธีคิดจากสมการเส้นตรงและ distance ไปเป็นชุดคำถามแบบ if/else:

```text
feature_2 <= 1.7 ?
├── yes → feature_0 <= -0.4 ?
│         ├── yes → class 0
│         └── no  → class 1
└── no  → class 1
```

Tree เป็นพื้นฐานของ Random Forest, Gradient Boosting, XGBoost, LightGBM และ CatBoost

## 2. Prerequisites

- classification metrics
- probability
- entropy concept
- NumPy
- train/validation/test
- overfitting / regularization

## 3. Learning Objectives

เมื่อจบบทนี้คุณควร:

- อธิบาย root / node / branch / leaf
- คำนวณ Gini impurity
- คำนวณ entropy และ information gain
- หา candidate thresholds
- implement greedy classification tree จากศูนย์
- อธิบาย recursive partitioning
- อธิบาย max_depth / min_samples_split / min_samples_leaf
- อธิบาย pruning concept
- อธิบาย why trees usually do not need feature scaling
- inspect feature importance limitations
- compare implementation กับ scikit-learn

## 4. Tree Mental Model

Tree แบ่ง feature space เป็น regions

```text
R² feature space
     ↓ split x₁ <= t
left region | right region
     ↓ split again
smaller regions
     ↓
leaf prediction
```

ใน classification แต่ละ leaf มักทำนาย majority class หรือ class probabilities จาก label distribution ใน leaf

## 5. Impurity

node ที่ pure:

```text
[0,0,0,0] → impurity ต่ำ
```

node ที่ mixed:

```text
[0,0,1,1] → impurity สูงกว่า
```

split ที่ดีทำให้ children บริสุทธิ์ขึ้น

## 6. Gini Impurity

ถ้า class proportions คือ `p₁,...,p_K`:

```text
Gini = 1 - Σ p_k²
```

binary 50/50:

```text
1 - 0.5² - 0.5² = 0.5
```

pure node:

```text
1 - 1² = 0
```

## 7. Entropy

```text
H = -Σ p_k log₂(p_k)
```

pure node → 0

binary 50/50 → 1 bit

## 8. Weighted Child Impurity

parent มี n samples

left มี `n_L`, right มี `n_R`

```text
I_children
=
(n_L/n) I(left)
+
(n_R/n) I(right)
```

## 9. Information Gain

```text
Gain
=
I(parent)
-
I_children
```

เลือก split ที่ gain สูงสุด

ถ้าใช้ Gini ก็เรียก impurity decrease ได้เช่นกัน

## 10. Candidate Thresholds

สำหรับ numeric feature:

```text
values = [1, 2, 5, 9]
```

candidate thresholds ที่มีเหตุผลคือ midpoint ระหว่าง unique sorted values:

```text
1.5, 3.5, 7.0
```

ไม่จำเป็นต้องลองทุก real number

## 11. Greedy Learning

Decision Tree แบบมาตรฐานสร้างแบบ greedy:

1. หา best split ที่ node ปัจจุบัน
2. แบ่ง data
3. recurse ซ้าย/ขวา
4. หยุดตาม stopping rules

greedy ไม่รับประกัน global optimal tree

## 12. Stopping Rules

สำคัญ:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- minimum impurity decrease
- pure node

ถ้าไม่ constrain tree สามารถ memorize training data ได้ง่าย

## 13. Overfitting

deep tree:

- training error ต่ำ
- boundary ซับซ้อน
- variance สูง

shallow tree:

- bias สูงขึ้น
- variance ลดลง

นี่คือเหตุผลที่ Random Forest aggregate trees หลายต้น

## 14. Pruning

สองแนวคิด:

### Pre-pruning
หยุดก่อน tree โตมาก:

- max_depth
- min_samples_leaf
- min_samples_split

### Post-pruning
ปลูก tree ก่อนแล้วตัด branches ที่ไม่คุ้ม complexity

scikit-learn รองรับ cost-complexity pruning ผ่าน `ccp_alpha`

## 15. Classification Probability

leaf มี:

```text
class 0: 8 samples
class 1: 2 samples
```

probability estimate แบบง่าย:

```text
P(0)=0.8
P(1)=0.2
```

แต่ probability จาก leaf อาจไม่ calibrated โดยเฉพาะ leaf เล็ก

## 16. Why Scaling Usually Not Needed

Tree ถาม:

```text
x_j <= threshold
```

monotonic rescaling เช่น:

```text
meters → centimeters
```

เปลี่ยน threshold แต่ไม่เปลี่ยน ordering ของ values

ต่างจาก KNN / linear regularization ที่ scale มีผลโดยตรง

## 17. Missing Values

การจัดการ missing values แตกต่างตาม implementation/library

อย่า assume ว่า tree ทุก library handle NaN เหมือนกัน

ใน from-scratch implementation ของบทนี้:
- NaN ถูก reject เพื่อให้ logic ชัด

ใน production ต้องกำหนด missing-value policy อย่าง explicit

## 18. Categorical Features

tree แบบพื้นฐานรับ numeric split

categorical options:

- one-hot encode
- ordinal encode อย่างระวัง
- native categorical split ในบาง libraries เช่น LightGBM/CatBoost/XGBoost รุ่นใหม่

Chapter 12 จะกลับมาเรื่องนี้

## 19. Feature Importance

Mean Decrease in Impurity (MDI) สรุป impurity decrease ที่ feature สร้าง

ข้อจำกัด:

- biased ต่อ high-cardinality / many-split features
- correlated features แบ่ง importance กัน
- importance ≠ causality

ใช้ permutation importance / SHAP ในบท interpretability ต่อไป

## 20. Complexity

naive exact split search implementation ของเราเน้นการเรียนรู้ ไม่ได้ optimize

production implementations ใช้ sorting/cache/histograms และ low-level optimization

## 21. From Scratch

[src/decision_tree.py](src/decision_tree.py)

ประกอบด้วย:

- `gini_impurity`
- threshold generation
- best split search
- recursive nodes
- predict
- predict_proba

## 22. scikit-learn

ใช้:

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=...,
    min_samples_leaf=...,
    random_state=...
)
```

สำคัญ: tune complexity บน validation/CV

## 23. Failure Cases

- noisy labels
- small dataset + deep tree
- unstable small data changes
- extrapolation ใน regression tree
- high-cardinality noise features
- leakage
- wrong validation protocol

## 24. Debugging

ถ้า tree แปลก:

- print node sample count
- print impurity
- print selected feature/threshold
- verify children non-empty
- inspect depth
- compare stump manually
- test pure node
- test constant feature

## 25. Common Mistakes

1. ใช้ train accuracy เลือก depth
2. ปล่อย tree เต็ม depth โดยไม่ validate
3. feature importance = causality
4. scale data โดยคิดว่าจำเป็นเสมอ
5. threshold candidates ผิด
6. allow empty child
7. recurse โดยไม่มี stopping condition
8. entropy log(0) โดยไม่ skip p=0
9. test set ใช้เลือก pruning
10. compare different splits

## 26. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 27. Checklist

- [ ] Gini
- [ ] Entropy
- [ ] weighted impurity
- [ ] information gain
- [ ] threshold generation
- [ ] recursion
- [ ] stopping rules
- [ ] pruning concept
- [ ] scaling reasoning
- [ ] feature-importance caveats

## 28. What's Next

Chapter 11 จะลด variance ของ single tree ด้วย **bagging + random feature subsets → Random Forest**
