# Batch 01 Integration Project — Mini Data Science Engine

โปรเจกต์นี้บังคับใช้ความรู้จากทั้ง 3 บทพร้อมกัน

```text
CLI / pathlib                 Chapter 01
      ↓
CSV input
      ↓
validation + cleaning         Chapters 01 + 03
      ↓
manual statistics            Chapter 02
      ↓
NumPy verification           Chapters 02 + 03
      ↓
Pandas analysis              Chapter 03
      ↓
Matplotlib plots             Chapter 03
      ↓
JSON + Markdown report
```

## Goal

สร้าง pipeline ที่รับ CSV แล้วสร้าง:

- schema/shape
- missing report
- duplicate count
- numeric statistics
- manual-vs-NumPy consistency checks
- correlation matrix
- histograms
- Markdown report
- JSON machine-readable report

## Run

จาก repository root:

```bash
source .venv/bin/activate

python integration-project/src/mini_data_science_engine.py \
  03-scientific-python/data/sample_students.csv \
  --output-dir reports/integration
```

ผลลัพธ์:

```text
reports/integration/
├── report.json
├── report.md
└── plots/
    ├── ...
```

## What you must understand

### Chapter 01

- `argparse`
- `pathlib`
- functions
- exceptions
- JSON I/O
- testable architecture

### Chapter 02

- mean
- population variance
- covariance/correlation interpretation
- floating-point approximate equality

### Chapter 03

- ndarray
- vectorized statistics
- DataFrame
- missing values
- correlation
- plotting

## Required extension before mastery

เพิ่มเอง:

1. CLI `--drop-duplicates`
2. CLI `--columns a b c`
3. detect constant columns
4. scatter plot สำหรับ numeric pairs ที่เลือก
5. Markdown section อธิบาย data-quality warnings
6. test dataset ที่มี column numeric ทั้งหมดเป็น missing

## Rules

- ห้ามแก้ raw input file
- cleaning rules ต้อง explicit
- report ต้องบอก input ที่ใช้
- observation ต้องแยกจาก causal claim
- ทุก bug fix ต้องเพิ่ม regression test ถ้าเหมาะสม

## Mastery challenge

หา CSV ใหม่ที่ไม่ใช่ sample file แล้วตอบ:

1. ข้อมูลมีปัญหาคุณภาพอะไร?
2. numeric distributions เป็นอย่างไร?
3. มี relationships อะไรที่น่าสำรวจต่อ?
4. อะไร “ยังสรุปไม่ได้” จากข้อมูลนี้?
5. ถ้าจะใช้สร้าง ML model ต้องเตรียมอะไรต่อ?

เมื่อทำได้ คุณพร้อมเข้าสู่ Chapter 04 — Data Fundamentals
