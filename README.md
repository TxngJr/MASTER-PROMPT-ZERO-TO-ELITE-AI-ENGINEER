# Zero to Elite AI Engineer

หลักสูตรภาษาไทยแบบ **from first principles** สำหรับการเรียน AI ตั้งแต่ Programming/Math ไปจนถึงสร้าง LLM, fine-tune, align, optimize และ deploy โดยใช้ Fedora Linux เป็นสภาพแวดล้อมหลัก

> เป้าหมายของ repository นี้ไม่ใช่การจำ API แต่คือ `เข้าใจ → derive → implement → debug → experiment → build`

## Target machine

หลักสูตรออกแบบให้ทำ Local Lab ได้บน laptop ระดับ:

- Fedora Linux
- AMD Ryzen 7 5825U
- NVIDIA RTX 3050 Ti Laptop GPU
- RAM 16 GB
- SSD 512 GB

งานที่ใหญ่เกินเครื่องจะมี **Local Version** และ **Scaled Version** แยกกันในบทที่เกี่ยวข้อง

## Completed curriculum

### Batch 01 — Foundations

1. [Chapter 01 — Programming Foundations](01-programming/README.md)
2. [Chapter 02 — Mathematics Foundations](02-mathematics/README.md)
3. [Chapter 03 — Scientific Python](03-scientific-python/README.md)

Integration:
[Mini Data Science Engine](integration-project/README.md)

### Batch 02 — Data, ML Fundamentals & Regression

4. [Chapter 04 — Data Fundamentals](04-data/README.md)
5. [Chapter 05 — Machine Learning Fundamentals](05-ml-fundamentals/README.md)
6. [Chapter 06 — Regression](06-regression/README.md)

Integration:
[Leakage-Safe Regression System](integration-project-batch02/README.md)

### Batch 03 — Classification Foundations

7. [Chapter 07 — Logistic Regression](07-logistic-regression/README.md)
8. [Chapter 08 — K-Nearest Neighbors](08-knn/README.md)
9. [Chapter 09 — Naive Bayes](09-naive-bayes/README.md)

Integration:
[Classification Model Lab](integration-project-batch03/README.md)

### Batch 04 — Trees & Ensembles

10. [Chapter 10 — Decision Trees](10-decision-tree/README.md)
11. [Chapter 11 — Random Forest](11-random-forest/README.md)
12. [Chapter 12 — Gradient Boosting](12-gradient-boosting/README.md)

Chapter 12 includes AdaBoost, classic Gradient Boosting, histogram boosting และ engineering concepts ของ XGBoost / LightGBM / CatBoost.

Integration:
[Tree Ensemble Benchmark](integration-project-batch04/README.md)

Optional boosting libraries:

```bash
python -m pip install -r requirements-batch04-extras.txt
```

### Batch 05 — Geometry, Clustering & Representation

13. [Chapter 13 — Support Vector Machines](13-svm/README.md)
14. [Chapter 14 — Clustering](14-clustering/README.md)
15. [Chapter 15 — Dimensionality Reduction](15-dimensionality-reduction/README.md)

Chapter 14 includes:

- K-Means
- K-Means++
- DBSCAN
- Hierarchical / Agglomerative Clustering
- Gaussian Mixture Models / EM

Chapter 15 includes:

- PCA
- SVD
- TruncatedSVD concepts
- t-SNE
- UMAP

Integration:
[Geometry & Representation Lab](integration-project-batch05/README.md)

Optional UMAP dependency:

```bash
python -m pip install -r requirements-batch05-extras.txt
```

## Learning loop

ทุกบทใช้วงจร:

```text
Theory
  ↓
Intuition
  ↓
Mathematics / Systems Model
  ↓
From-Scratch Implementation
  ↓
Library Implementation
  ↓
Experiment
  ↓
Debug
  ↓
Project
  ↓
Review
```

อย่าข้าม exercises และอย่าอ่าน solutions ก่อนพยายามทำเอง

## Setup

อ่าน [SETUP_FEDORA.md](SETUP_FEDORA.md) ก่อน

สร้าง environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

สำหรับเนื้อหาปัจจุบันถึง Chapter 15:

```bash
python -m pip install -r requirements-batch05.txt
```

รัน tests:

```bash
pytest -q
```

## Recommended study order

```text
01 Programming
   ↓
02 Mathematics
   ↓
03 Scientific Python
   ↓
Batch 01 Integration
   ↓
04 Data
   ↓
05 ML Fundamentals
   ↓
06 Regression
   ↓
Batch 02 Integration
   ↓
07 Logistic Regression
   ↓
08 KNN
   ↓
09 Naive Bayes
   ↓
Batch 03 Integration
   ↓
10 Decision Trees
   ↓
11 Random Forest
   ↓
12 Gradient Boosting
   ↓
Batch 04 Integration
   ↓
13 Support Vector Machines
   ↓
14 Clustering
   ↓
15 Dimensionality Reduction
   ↓
Batch 05 Integration
```

## Reviews

- [BATCH_REVIEW.md](BATCH_REVIEW.md) — Batch 01
- [BATCH_02_REVIEW.md](BATCH_02_REVIEW.md) — Batch 02
- [BATCH_03_REVIEW.md](BATCH_03_REVIEW.md) — Batch 03
- [BATCH_04_REVIEW.md](BATCH_04_REVIEW.md) — Batch 04
- [BATCH_05_REVIEW.md](BATCH_05_REVIEW.md) — Batch 05

## Definition of mastery

บทหนึ่งถือว่า “เข้าใจ” เมื่อทำได้อย่างน้อย:

1. **Explain** — อธิบายด้วยคำของตัวเอง
2. **Derive** — ไล่ที่มาของสมการสำคัญ
3. **Implement** — เขียนแก่น algorithm
4. **Debug** — หา root cause
5. **Modify** — เปลี่ยนโจทย์แล้วปรับ implementation
6. **Apply** — ใช้กับข้อมูลใหม่

## Current status

- [x] Batch 01 — Chapters 01–03
- [x] Batch 02 — Chapters 04–06
- [x] Batch 03 — Chapters 07–09
- [x] Batch 04 — Chapters 10–12
- [x] Batch 05 — Chapters 13–15
- [ ] Batch 06 — Chapters 16–18
- [ ] Batch 07+ — รอ batch ก่อนหน้าผ่าน quality audit

บทถัดไป:

- Chapter 16 — Anomaly Detection
- Chapter 17 — Time Series
- Chapter 18 — Neural Network Foundations
