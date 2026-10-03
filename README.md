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

งานที่ใหญ่เกินเครื่องจะมีแนวคิด **Local Version** และ **Scaled Version** แยกกันในบทที่เกี่ยวข้อง

## Batch 01

Batch แรกมี 3 บท:

1. [Chapter 01 — Programming Foundations](01-programming/README.md)
2. [Chapter 02 — Mathematics Foundations](02-mathematics/README.md)
3. [Chapter 03 — Scientific Python](03-scientific-python/README.md)

และปิดท้ายด้วย [Mini Data Science Engine](integration-project/README.md) ที่รวม Python + Linux/Git + คณิตศาสตร์ + NumPy/Pandas/Matplotlib เข้าด้วยกัน

## วิธีเรียน

ทุกบทใช้วงจรเดียวกัน:

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

อย่าข้าม exercise และอย่าอ่านเฉพาะ solution เพราะเป้าหมายคือสามารถอธิบายและเขียนใหม่ได้เอง

## เริ่มต้น

อ่าน [SETUP_FEDORA.md](SETUP_FEDORA.md) ก่อน แล้วสร้าง environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-batch01.txt
```

ตรวจ environment:

```bash
python --version
git --version
gcc --version
```

รัน test ทั้ง Batch:

```bash
pytest -q
```

## Foundation references

- [COURSE_MAP.md](COURSE_MAP.md) — แผน 80 บท
- [GLOSSARY.md](GLOSSARY.md) — ศัพท์สำคัญ
- [MATH_REFERENCE.md](MATH_REFERENCE.md) — สูตรที่ใช้ซ้ำ
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — วิธี debug environment
- [BATCH_REVIEW.md](BATCH_REVIEW.md) — checklist และ audit ของ Batch 01

## Definition of mastery

บทหนึ่งถือว่า “เข้าใจ” เมื่อทำได้อย่างน้อย 6 อย่าง:

1. **Explain** — อธิบายแนวคิดด้วยคำของตัวเอง
2. **Derive** — ไล่ที่มาของสมการสำคัญ
3. **Implement** — เขียนแก่นของ algorithm ได้
4. **Debug** — หา bug จากข้อมูล/shape/type/environment ได้
5. **Modify** — เปลี่ยนโจทย์หรือข้อกำหนดแล้วปรับ code ได้
6. **Apply** — นำไปใช้กับข้อมูลใหม่ได้

## Current status

- [x] Batch 01 — Chapters 01–03
- [ ] Batch 02 — Chapters 04–06
- [ ] Batch 03+ — รอ Batch ก่อนหน้าผ่าน quality audit

Chapter 04 จะไม่ถูกสร้างก่อน Batch 01 ผ่านการตรวจ prerequisite, code, math และ project integration
