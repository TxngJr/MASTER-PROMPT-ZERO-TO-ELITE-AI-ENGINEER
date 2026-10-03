# Chapter 03 — Scientific Python: NumPy, Pandas, Matplotlib

## 1. Why This Matters

Python loop เข้าใจง่ายแต่ช้าเมื่อทำงานตัวเลขจำนวนมาก Scientific Python ย้าย computation ไปยัง optimized array kernels และให้เครื่องมือจัดการตารางข้อมูล/visualization

```text
Python values
   ↓
NumPy arrays
   ↓
vectorized math
   ↓
Pandas tables
   ↓
analysis/cleaning
   ↓
Matplotlib
   ↓
visual evidence
```

## 2. Prerequisites

จาก Chapters 01–02:

- Python functions/classes/modules
- vector/matrix
- mean/variance
- basic calculus/statistics

## 3. Learning Objectives

By the end of this chapter you can:

- สร้าง NumPy ndarray และอธิบาย shape/dtype/ndim
- index/slice/mask arrays
- ใช้ broadcasting อย่างถูกต้อง
- ใช้ vectorization แทน Python loops
- เข้าใจ axis ใน reductions
- อ่าน CSV ด้วย Pandas
- inspect schema, missing values, duplicates
- select/filter/group/aggregate
- แปลง dtype อย่างระวัง
- สร้าง histogram/scatter/line plot
- แยก exploratory notebook ออกจาก reusable Python functions
- สร้าง reproducible analysis pipeline

## 4. Study Order

1. [NumPy](theory/01-numpy.md)
2. [Pandas](theory/02-pandas.md)
3. [Matplotlib & Jupyter](theory/03-matplotlib-jupyter.md)
4. [Dataset Analyzer](src/analyze_dataset.py)
5. [Exercises](exercises/README.md)
6. [Mini Project](mini-project/README.md)

## 5. NumPy Array Mental Model

```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])

print(x.shape)  # (2, 2)
print(x.ndim)   # 2
print(x.dtype)
```

Array ไม่ใช่ nested Python list ธรรมดา มันมี homogeneous dtype และ memory-oriented representation ที่ optimized กว่า

## 6. Shape

```text
(100,)      vector-like 1D array
(100, 4)   100 rows × 4 columns
(32, 3, 224, 224) batch of images in NCHW convention
```

shape จะกลายเป็น debugging skill ที่ใช้ทุกวันใน deep learning

## 7. Vectorization

Python:

```python
result = []
for value in values:
    result.append(value * 2)
```

NumPy:

```python
result = values * 2
```

Vectorized code มักเร็วกว่าเพราะ loop หลักเกิดใน compiled implementation และ memory access เหมาะกับ numeric workloads มากกว่า

## 8. Broadcasting

```python
x = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])

print(x + bias)
```

shape:

```text
x    (2,3)
bias   (3,)
→    (2,3)
```

NumPy align dimensions จากด้านขวา; dimensions compatible เมื่อเท่ากันหรือหนึ่งด้านเป็น 1

## 9. Axis

```python
x.mean(axis=0)  # reduce rows → one value per column
x.mean(axis=1)  # reduce columns → one value per row
```

อย่าจำว่า axis=0 = column “ตลอดไป” ให้คิดว่า **axis ที่ระบุคือ axis ที่ถูก reduce ออก**

## 10. Pandas DataFrame

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
```

DataFrame เพิ่ม labels/index และ mixed dtypes เหมาะกับ tabular data

## 11. Selection

```python
scores = df["score"]
high_scores = df.loc[df["score"] >= 80, ["name", "score"]]
```

`.loc` ใช้ labels/boolean mask  
`.iloc` ใช้ integer positions

## 12. Missing Values

ห้าม drop ทุก missing row โดยอัตโนมัติ

ถามก่อน:

- ทำไม missing?
- missing แบบสุ่มหรือมี pattern?
- feature สำคัญแค่ไหน?
- impute ได้ไหม?
- dropping สร้าง bias หรือไม่?

Batch 02 จะลง data leakage/preprocessing ลึกขึ้น

## 13. Grouping

```python
df.groupby("group")["score"].agg(["count", "mean", "std"])
```

การ aggregate ทำให้เห็น structure ที่ row-by-row inspection มองไม่เห็น

## 14. Matplotlib

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(df["score"].dropna(), bins=10)
ax.set_xlabel("Score")
ax.set_ylabel("Count")
ax.set_title("Score distribution")
fig.tight_layout()
plt.show()
```

Plot ต้องมี title/axes/units ที่ตีความได้

## 15. Jupyter

Notebook เหมาะกับ exploration แต่มี hidden state risk

กฎ:

1. restart kernel
2. run all top-to-bottom
3. move reusable logic ไป `.py`
4. อย่าให้ cell order เป็น dependency ลับ

## 16. Performance

วัดก่อน optimize:

```python
import time

start = time.perf_counter()
# operation
elapsed = time.perf_counter() - start
```

อย่า conclude จาก run เดียวกับ input เล็กมาก

## 17. Memory

Array ขนาด:

```text
elements × bytes_per_element
```

เช่น float32 1,000,000 elements ≈ 4 MB raw storage

ดูจริง:

```python
print(x.nbytes)
```

concept นี้จะกลับมาเมื่อคำนวณ VRAM ของ model

## 18. Data Quality Debugging

ตรวจอย่างน้อย:

```python
print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isna().sum())
print(df.duplicated().sum())
```

อย่าเริ่ม model ก่อนเข้าใจ data

## 19. Production Perspective

Notebook ไม่ใช่ pipeline production โดยตัวมันเอง

แยก:
- I/O
- validation
- transforms
- metrics
- plots
- report

เป็น functions ที่ test ได้

## 20. Research Perspective

Visualization และ statistics ใช้ตรวจ assumptions ก่อน experiment เช่น skew, outliers, class imbalance, data collection artifact

## 21. Common Mistakes

1. broadcasting โดยไม่ตั้งใจ
2. axis ผิด
3. dtype integer ทำให้ operation ได้ผลที่ไม่คาด
4. chained assignment ใน Pandas
5. drop missing แบบไม่วิเคราะห์
6. duplicate rows ไม่ตรวจ
7. plot โดยไม่มี units/labels
8. เชื่อ correlation plot ว่าเป็น causation
9. notebook run ไม่เรียงลำดับ
10. แปลงทุกอย่างเป็น object/string
11. Python loop ทั้งที่ vectorize ได้
12. copy array/DataFrame ใหญ่โดยไม่จำเป็น

## 22. Interview Questions

1. ndarray ต่างจาก list อย่างไร?
2. broadcasting คืออะไร?
3. axis ใน reduction คืออะไร?
4. vectorization เร็วขึ้นเพราะอะไร?
5. view กับ copy สำคัญอย่างไร?
6. DataFrame กับ ndarray ใช้ต่างกันเมื่อไร?
7. missing values ควรจัดการอย่างไร?
8. `.loc` กับ `.iloc` ต่างกันอย่างไร?
9. ทำไม notebook hidden state อันตราย?
10. histogram กับ scatter plot ตอบคำถามต่างกันอย่างไร?

## 23. Checklist

- [ ] อธิบาย shape/ndim/dtype
- [ ] index/slice/mask
- [ ] broadcast ด้วย reasoning
- [ ] reductions หลาย axis
- [ ] load/inspect CSV
- [ ] clean missing/duplicates
- [ ] group/aggregate
- [ ] plot histogram/scatter
- [ ] run tests
- [ ] วิเคราะห์ sample dataset โดยไม่ดู solution

## 24. What's Next

Batch 02 จะเริ่ม Data Fundamentals → ML Fundamentals → Regression โดยใช้ skills จากทั้งสามบทนี้เป็น prerequisite
