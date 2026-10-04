# Zero to Elite AI Engineer

หลักสูตรภาษาไทยแบบ **from first principles** สำหรับการเรียน AI ตั้งแต่ Programming/Math ไปจนถึงสร้าง LLM, fine-tune, align, optimize และ deploy โดยใช้ Fedora Linux เป็นสภาพแวดล้อมหลัก

> เป้าหมายของ repository นี้ไม่ใช่การจำ API แต่คือ เข้าใจ → derive → implement → debug → experiment → build

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

Integration:
[Tree Ensemble Benchmark](integration-project-batch04/README.md)

Optional boosting libraries:

~~~bash
python -m pip install -r requirements-batch04-extras.txt
~~~

### Batch 05 — Geometry, Clustering & Representation

13. [Chapter 13 — Support Vector Machines](13-svm/README.md)
14. [Chapter 14 — Clustering](14-clustering/README.md)
15. [Chapter 15 — Dimensionality Reduction](15-dimensionality-reduction/README.md)

Integration:
[Geometry & Representation Lab](integration-project-batch05/README.md)

Optional UMAP:

~~~bash
python -m pip install -r requirements-batch05-extras.txt
~~~

### Batch 06 — Anomalies, Time & Neural Foundations

16. [Chapter 16 — Anomaly Detection](16-anomaly-detection/README.md)
17. [Chapter 17 — Time Series Forecasting](17-time-series/README.md)
18. [Chapter 18 — Neural Network Foundations](18-neural-network-foundations/README.md)

Integration:
[Anomaly-Aware Forecasting & Neural Baseline Lab](integration-project-batch06/README.md)

Optional Prophet:

~~~bash
python -m pip install -r requirements-batch06-extras.txt
~~~

### Batch 07 — Backpropagation, Optimizers & Stable Objectives

19. [Chapter 19 — Backpropagation & Reverse-Mode Autodiff](19-backpropagation/README.md)
20. [Chapter 20 — Optimizers](20-optimizers/README.md)
21. [Chapter 21 — Activations & Loss Functions](21-activations-losses/README.md)

Integration:
[Tiny Deep Learning Framework](integration-project-batch07/README.md)

This batch trains a nonlinear MLP using the repository's own Tensor, backward engine, AdamW and stable objectives before using full frameworks.

### Batch 08 — Frameworks & Convolution

22. [Chapter 22 — PyTorch](22-pytorch/README.md)
23. [Chapter 23 — TensorFlow / Keras](23-tensorflow-keras/README.md)
24. [Chapter 24 — Convolutional Neural Networks](24-cnn/README.md)

Chapter 22 maps the from-scratch engine to:
- torch.Tensor
- autograd
- nn.Module
- DataLoader
- torch.optim
- state_dict
- CPU / CUDA devices

Chapter 23 covers:
- tf.Tensor / tf.Variable
- GradientTape
- Sequential / Functional / subclassed Keras models
- compile / fit / custom loops
- tf.data
- Keras saving
- TensorFlow GPU visibility

Chapter 24 covers:
- convolution / cross-correlation
- channels / filters
- output-shape equations
- padding / stride / dilation
- receptive fields
- pooling
- NCHW / NHWC
- PyTorch Conv2d
- Keras Conv2D
- NumPy convolution from scratch

Integration:
[Cross-Framework CNN Lab](integration-project-batch08/README.md)

The integration uses the same sklearn digits data and split for both frameworks.

## Framework Installation

Core course dependencies through Batch 08:

~~~bash
python -m pip install -r requirements-batch08.txt
~~~

PyTorch is intentionally not pinned to a stale CUDA wheel.

For the Fedora + NVIDIA machine, use the current official PyTorch selector:
https://pytorch.org/get-started/locally/

TensorFlow CPU/simple environment:

~~~bash
python -m pip install -r requirements-batch08-tensorflow.txt
~~~

For TensorFlow GPU on Linux, follow the current official TensorFlow pip GPU instructions.

The heavy frameworks are isolated from the core requirements so future NumPy/scikit-learn tests do not repeatedly install multi-hundred-megabyte framework stacks.

## Learning loop

ทุกบทใช้วงจร:

~~~text
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
~~~

อย่าข้าม exercises และอย่าอ่าน solutions ก่อนพยายามทำเอง

## Setup

อ่าน [SETUP_FEDORA.md](SETUP_FEDORA.md) ก่อน

สร้าง environment:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-batch08.txt
~~~

Core tests:

~~~bash
pytest -q
~~~

Framework-specific tests require the corresponding framework environment.

## Recommended study order

~~~text
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
↓
16 Anomaly Detection
↓
17 Time Series
↓
18 Neural Network Foundations
↓
Batch 06 Integration
↓
19 Backpropagation
↓
20 Optimizers
↓
21 Activations & Loss Functions
↓
Batch 07 Tiny Deep Learning Framework
↓
22 PyTorch
↓
23 TensorFlow / Keras
↓
24 CNN
↓
Batch 08 Cross-Framework CNN Lab
~~~

## Reviews

- [BATCH_REVIEW.md](BATCH_REVIEW.md) — Batch 01
- [BATCH_02_REVIEW.md](BATCH_02_REVIEW.md) — Batch 02
- [BATCH_03_REVIEW.md](BATCH_03_REVIEW.md) — Batch 03
- [BATCH_04_REVIEW.md](BATCH_04_REVIEW.md) — Batch 04
- [BATCH_05_REVIEW.md](BATCH_05_REVIEW.md) — Batch 05
- [BATCH_06_REVIEW.md](BATCH_06_REVIEW.md) — Batch 06
- [BATCH_07_REVIEW.md](BATCH_07_REVIEW.md) — Batch 07
- [BATCH_08_REVIEW.md](BATCH_08_REVIEW.md) — Batch 08

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
- [x] Batch 06 — Chapters 16–18
- [x] Batch 07 — Chapters 19–21
- [x] Batch 08 — Chapters 22–24
- [ ] Batch 09 — Chapters 25–27
- [ ] Batch 10+ — รอ batch ก่อนหน้าผ่าน quality audit

บทถัดไป:

- Chapter 25 — Recurrent Neural Networks
- Chapter 26 — LSTM / GRU
- Chapter 27 — Autoencoders / Variational Autoencoders
