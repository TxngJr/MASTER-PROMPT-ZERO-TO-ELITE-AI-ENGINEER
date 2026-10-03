# Chapter 01 Exercises

ทำก่อนเปิด solutions

## Level 1 — Recall

1. variable name ต่างจาก object อย่างไร?
2. mutable กับ immutable ต่างกันอย่างไร?
3. `==` กับ `is` ต่างกันอย่างไร?
4. virtual environment มีประโยชน์อะไร?
5. Git commit เก็บอะไร?

## Level 2 — Understanding

6. ทำไม mutable default argument จึงเป็น bug ได้?
7. generator ช่วยเรื่อง memory อย่างไร?
8. ทำไม `except Exception: pass` อันตราย?
9. อธิบาย PATH ด้วยคำของตัวเอง
10. program กับ process ต่างกันอย่างไร?

## Level 3 — Coding

11. เขียน `safe_mean(values)` ที่ reject empty list
12. เขียน generator ที่ yield เลขคู่ 0..N
13. ใช้ `pathlib` หาไฟล์ `.csv` ทั้งหมดใต้ directory
14. เพิ่ม median ลง Dataset Inspector โดยไม่ใช้ pandas
15. เพิ่ม flag `--limit` เพื่ออ่านเพียง N rows

## Level 4 — Debugging

16. หา bug ใน:
```python
def append_score(score, scores=[]):
    scores.append(score)
    return scores
```

17. ทำไม `python -m pip` มักปลอดภัยกว่าเรียก `pip` ตรง ๆ เมื่อมีหลาย interpreter?
18. โปรแกรมอ่านไฟล์ได้ใน terminal A แต่ไม่ได้ใน terminal B — ระบุ state อย่างน้อย 4 อย่างที่ต้องตรวจ

## Level 5 — Challenge

19. เปลี่ยน inspector ให้ stream CSV แล้วคำนวณ row count/missing count โดยไม่เก็บทุก row
20. เพิ่ม type inference สำหรับ boolean และเขียน tests ป้องกัน regression
