# Mini Project — Data Quality & Preprocessing Pipeline

ใช้ dataset จาก Chapter 03 หรือ CSV ของตัวเอง

## Required

สร้าง CLI ที่:

1. load CSV
2. validate required columns
3. รายงาน dtypes/missing/duplicates
4. split train/validation/test
5. fit median imputer เฉพาะ train
6. fit standardizer เฉพาะ train
7. transform ทั้งสาม splits
8. save preprocessing metadata
9. save Markdown data-quality report
10. มี tests ป้องกัน leakage

## Leakage test ที่ต้องมี

สร้าง test data ที่ test set มีค่า extreme เช่น 10,000

ถ้า scaler mean เปลี่ยนตาม test value แปลว่า implementation รั่วข้อมูล

## Deliverables

```text
mini-project/
├── src/
├── tests/
└── REPORT.md
```

ใน REPORT ต้องอธิบายว่า feature ใดไม่ควรถูกใช้และเพราะอะไร
