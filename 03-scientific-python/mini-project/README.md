# Mini Project — Analyze and Visualize a Dataset

ใช้ [../data/sample_students.csv](../data/sample_students.csv) ก่อน แล้วเปลี่ยนเป็น dataset อื่น

## Required analysis

1. shape/dtypes
2. missing values
3. duplicates
4. numeric descriptive statistics
5. group-level summary
6. correlation matrix ของ numeric columns
7. histogram อย่างน้อย 2 plots
8. scatter plot อย่างน้อย 1 plot
9. สรุป observations
10. แยก observation ออกจาก causal claim

## Run starter

```bash
python 03-scientific-python/src/analyze_dataset.py \
  03-scientific-python/data/sample_students.csv
```

## Required extensions

- เพิ่ม correlation table
- เพิ่ม scatter `study_hours vs score`
- เพิ่ม CLI option เลือก columns
- save Markdown report
- test invalid/missing inputs

## Reproducibility

report ต้องระบุ:
- input path
- row/column counts
- cleaning rules
- generated files
- command ที่ใช้รัน
