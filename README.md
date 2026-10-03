# Zero to Elite AI Engineer

หลักสูตรภาษาไทยแบบ **from first principles** สำหรับการเรียน AI ตั้งแต่พื้นฐาน Programming/Math ไปจนถึงสร้าง LLM, fine-tune, align, optimize และ deploy โดยใช้ Fedora Linux เป็นสภาพแวดล้อมหลัก

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

สำหรับเนื้อหาปัจจุบันถึง Chapter 06:

```bash
python -m pip install -r requirements-batch02.txt
```

ตรวจ:

```bash
python --version
git --version
gcc --version
```

รัน tests ทั้ง repository:

```bash
pytest -q
```

## Recommended study order

```text
01-programming
    ↓
02-mathematics
    ↓
03-scientific-python
    ↓
Batch 01 Integration Project
    ↓
04-data
    ↓
05-ml-fundamentals
    ↓
06-regression
    ↓
Batch 02 Integration Project
```

## Foundation references

- [COURSE_MAP.md](COURSE_MAP.md) — แผน 80 บท
- [GLOSSARY.md](GLOSSARY.md) — ศัพท์สำคัญ
- [MATH_REFERENCE.md](MATH_REFERENCE.md) — สูตรที่ใช้ซ้ำ
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — วิธี debug environment
- [BATCH_REVIEW.md](BATCH_REVIEW.md) — audit Batch 01
- [BATCH_02_REVIEW.md](BATCH_02_REVIEW.md) — audit Batch 02

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
- [ ] Batch 03 — Chapters 07–09
- [ ] Batch 04+ — รอ batch ก่อนหน้าผ่าน quality audit

ถ้า Batch 02 CI และ mastery gate ผ่าน บทถัดไปคือ:

- Chapter 07 — Logistic Regression
- Chapter 08 — K-Nearest Neighbors
- Chapter 09 — Naive Bayes
