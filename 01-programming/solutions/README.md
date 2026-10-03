# Chapter 01 Solutions

อย่าใช้ไฟล์นี้ก่อนพยายามทำเอง

## Concepts

1. name เป็น identifier; object คือ value/entity ที่ name อ้าง
2. mutable เปลี่ยน state ภายใน object เดิมได้
3. `==` เปรียบเทียบ equality; `is` เปรียบเทียบ identity
4. venv แยก dependencies ต่อ project
5. commit บันทึก snapshot/tree + metadata + parent relationship

## Key reasoning

6. default object ถูกสร้างครั้งเดียว จึงแชร์ state ข้าม calls ได้
7. generator ผลิตค่าตามต้องการ ไม่ต้อง materialize ทั้ง sequence
8. broad silent catch ซ่อน root cause
9. PATH คือ search path ของ executable
10. program คือไฟล์/code; process คือ execution instance

## Sample coding solution

```python
def safe_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
```

```python
def even_numbers(limit: int):
    for value in range(limit + 1):
        if value % 2 == 0:
            yield value
```

```python
from pathlib import Path

csv_files = list(Path(".").rglob("*.csv"))
```

## Debugging answers

16. เปลี่ยน default เป็น `None` แล้วสร้าง list ภายใน
17. `python -m pip` ผูก pip กับ interpreter ที่เราเลือกชัดเจน
18. ตรวจ `pwd`, environment variables, user/permissions, active venv, path, arguments และ file existence

ข้อ 14–15 และ challenge ควรแก้ใน code จริงและเพิ่ม test ไม่ควร copy solution สำเร็จรูป เพราะเป็นโจทย์ design
