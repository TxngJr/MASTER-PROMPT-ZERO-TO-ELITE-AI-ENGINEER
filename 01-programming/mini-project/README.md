# Mini Project — CLI Dataset Inspector

## Goal

สร้าง CLI ที่มอง CSV ครั้งแรกแล้วตอบได้ว่า:

- มีกี่ rows/columns
- column ชื่ออะไร
- type คร่าว ๆ คืออะไร
- missing เท่าไร
- unique values เท่าไร
- numeric min/max/mean เท่าไร

Starter implementation อยู่ที่ [../src/dataset_inspector.py](../src/dataset_inspector.py)

## Run

```bash
python 01-programming/src/dataset_inspector.py 03-scientific-python/data/sample_students.csv
```

## Required extension

เพิ่มอย่างน้อย 3 อย่าง:

1. `--limit N`
2. median สำหรับ numeric columns
3. JSON output file ผ่าน `--output report.json`

## Engineering constraints

- standard library only
- ห้าม pandas ใน project นี้
- ใช้ `pathlib`
- error message ต้องอ่านรู้เรื่อง
- core logic ต้องแยกจาก CLI
- tests ต้องครอบคลุม happy path + invalid path + missing values

## Stretch

ทำ streaming version ที่ memory usage ไม่โตตามจำนวน rows ทั้งไฟล์
