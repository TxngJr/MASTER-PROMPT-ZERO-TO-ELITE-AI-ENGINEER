# Chapter 16 — Anomaly Detection

## 1. Why This Matters

Anomaly Detection ถามคำถามต่างจาก classification:

~~~text
ข้อมูลนี้ผิดปกติแค่ไหน?
~~~

ไม่จำเป็นต้องมี anomaly labels ครบทุกตัว และในงานจริง anomalies มักหายากมาก

ตัวอย่าง:
- sensor failure
- fraudulent-looking behavior
- manufacturing defects
- server metrics spike
- unusual user activity
- data pipeline corruption

## 2. Outlier Detection vs Novelty Detection

**Outlier Detection**
training data อาจมี anomalies ปนอยู่แล้ว

**Novelty Detection**
training data สมมติว่า mostly normal แล้วตรวจ sample ใหม่ภายหลัง

ความต่างนี้สำคัญมาก เช่น Local Outlier Factor ใน scikit-learn ใช้ novelty=False สำหรับ outlier detection โดย default; ถ้าจะ score unseen data ต้องใช้ novelty=True

## 3. Learning Objectives

เมื่อจบบทนี้คุณควร:
- แยก anomaly / outlier / novelty
- ใช้ robust univariate scores
- อธิบาย multivariate anomaly
- implement z-score / robust-z / Mahalanobis detector
- อธิบาย Isolation Forest
- อธิบาย Local Outlier Factor
- อธิบาย One-Class SVM
- tune threshold ด้วย validation labels เมื่อมี
- ประเมินด้วย precision/recall/PR-AUC
- หลีกเลี่ยง contamination leakage
- เข้าใจ time-dependent anomaly caveats

## 4. Univariate Z-Score

~~~text
z = (x - μ) / σ
~~~

ถ้า |z| สูง sample อาจผิดปกติ

ข้อจำกัด:
- mean/std sensitive ต่อ outliers
- Gaussian-like assumption โดยนัย
- ไม่ capture feature interactions

## 5. Robust Z-Score

ใช้ median และ MAD:

~~~text
MAD = median(|x - median(x)|)
~~~

modified score:

~~~text
z_robust ≈ 0.6745(x - median)/MAD
~~~

robust ต่อ extreme values มากกว่า mean/std

## 6. Mahalanobis Distance

sample อาจดูปกติทีละ feature แต่ผิดปกติเมื่อมองร่วมกัน

~~~text
D²(x)
=
(x-μ)ᵀ Σ⁻¹ (x-μ)
~~~

มัน account covariance ระหว่าง features

ข้อจำกัด:
- covariance invertibility
- sensitive เมื่อ d ใหญ่เทียบ n
- classical mean/covariance ไม่ robust ต่อ contamination

## 7. Isolation Forest Intuition

Isolation Forest ถามว่า point นี้ถูก isolate ด้วย random splits ได้เร็วแค่ไหน

algorithm:
1. sample data
2. choose random feature
3. choose random split
4. recurse
5. measure path length
6. average across trees

shorter path โดยทั่วไปหมายถึง anomalous มากขึ้น

## 8. Contamination

contamination เป็น assumption/thresholding aid ว่า anomaly fraction ประมาณเท่าไร

ห้าม:
- ตั้ง contamination จาก test labels แล้วรายงาน test เดิม
- assume anomaly fraction คงที่ตลอด production

threshold ควรแยกจาก raw anomaly score

## 9. Local Outlier Factor

LOF เปรียบเทียบ local density ของ point กับ neighbors

~~~text
density(point)
vs
density(neighbors)
~~~

ถ้า point มี density ต่ำกว่าบริเวณรอบ ๆ มาก อาจเป็น local outlier

ข้อเสีย:
- sensitive ต่อ k / metric / scaling
- high-dimensional neighborhoods เสื่อม

## 10. One-Class SVM

พยายาม estimate support/boundary ของ normal distribution ใน feature space

RBF kernel ช่วยสร้าง nonlinear boundary

parameters สำคัญ:
- nu
- gamma
- kernel

ต้อง scale features

## 11. Global vs Local Anomaly

point อาจ global-far แต่ local-normal หรือกลับกัน

ดังนั้น anomaly definition ขึ้นกับ context และ representation

## 12. Supervised Evaluation

ถ้ามี anomaly labels:
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

สำหรับ rare anomalies PR-AUC มัก informative กว่า accuracy

## 13. Threshold Selection

anomaly score:

~~~text
higher score → more abnormal
~~~

เลือก threshold บน validation set ตาม cost

เช่น:
- false negative แพง → favor recall
- false positive แพง → favor precision

## 14. Unsupervised Evaluation Problem

ถ้าไม่มี labels จริง:
- score distribution
- stability
- domain inspection
- synthetic corruption
- human review
- downstream impact

ไม่มี internal metric universal ที่พิสูจน์ว่า detector ถูกต้อง

## 15. Time-Series Anomalies

time series ต้องพิจารณา:
- point anomaly
- contextual anomaly
- collective anomaly

ค่าเดียวกันอาจปกติกลางวันแต่ผิดปกติตอนกลางคืน

## 16. Leakage

ถ้า fit scaler / covariance / anomaly detector บน full data ก่อน split:

~~~text
future/test distribution
→ influence detector
~~~

นี่คือ leakage

สำหรับ novelty detection:
- fit เฉพาะ training period
- score validation/test ใหม่

## 17. From Scratch

ดู src/anomaly.py

มี:
- ZScoreDetector
- RobustZScoreDetector
- MahalanobisDetector
- threshold helper

## 18. scikit-learn

สำคัญ:
- IsolationForest
- LocalOutlierFactor
- OneClassSVM

LOF:
- novelty=False → training-set outlier detection
- novelty=True → score/predict unseen samples

## 19. Failure Cases

- unscaled distances
- anomaly distribution changes
- contamination wrong
- high-dimensional sparse data
- local-density assumption wrong
- normal training set already contaminated
- covariance singular
- time leakage

## 20. Common Mistakes

1. ใช้ accuracy อย่างเดียว
2. remove every anomaly automatically
3. fit detector before split
4. anomaly score = probability
5. threshold tuned on test
6. LOF novelty mode misunderstood
7. One-Class SVM unscaled
8. anomaly = invalid data เสมอ
9. rare but valuable events deleted
10. synthetic anomalies treated as perfect ground truth

## 21. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 22. Checklist

- [ ] outlier vs novelty
- [ ] z / robust-z
- [ ] Mahalanobis
- [ ] Isolation Forest
- [ ] LOF
- [ ] One-Class SVM
- [ ] threshold
- [ ] PR-AUC
- [ ] leakage
- [ ] contextual anomalies

## 23. What's Next

Chapter 17 จะเพิ่มแกนเวลา ทำให้ random split และ IID assumptions ใช้ไม่ได้ตรง ๆ
