# Chapter 01 — Programming Foundations

## 1. Why This Matters

AI engineering คือ software engineering ที่มีข้อมูลและคณิตศาสตร์เข้ามาเพิ่ม ถ้า Python, Linux, Git และ debugging ยังไม่แน่น เราจะเสียเวลาแก้ environment มากกว่าเรียน model

บทนี้มีเป้าหมายให้คุณ “เขียนโปรแกรมเพื่อทดลอง” ได้ ไม่ใช่แค่จำ syntax

## 2. Prerequisites

- ใช้ terminal ได้ในระดับเปิดและพิมพ์คำสั่ง
- ไม่ต้องมีพื้นฐาน Python มาก่อน

## 3. Learning Objectives

By the end of this chapter you can:

- อธิบาย variable, value, type, object และ reference
- ใช้ condition, loop, function, class, iterator, exception
- เขียน type hints และแบ่ง code เป็น module
- ใช้ standard library สำคัญ เช่น `pathlib`, `csv`, `argparse`
- ใช้ Bash สำรวจ filesystem/process/environment
- ใช้ Git แบบ branch → diff → commit
- อธิบาย stack/heap/pointer ในระดับ mental model
- debug โปรแกรมจาก traceback และ regression test
- สร้าง CLI Dataset Inspector

## 4. Mental Model

```text
source code
   ↓
Python interpreter
   ↓
objects + references
   ↓
OS process
   ↓
CPU / memory / files
```

โปรแกรม AI ในอนาคตก็ยังอยู่บนฐานเดียวกัน เพียงมี array/tensor/GPU เพิ่มเข้ามา

## 5. Study Order

1. [Python Foundations](theory/01-python-foundations.md)
2. [Linux, Bash, Git, C/C++ Memory Model](theory/02-linux-git-memory.md)
3. อ่านและรัน [Dataset Inspector](src/dataset_inspector.py)
4. ทำ [Exercises](exercises/README.md)
5. ตรวจด้วย [Solutions](solutions/README.md)
6. ทำ [Mini Project](mini-project/README.md)

## 6. First Python Program

```python
message: str = "AI starts with fundamentals."
print(message)
```

`message` คือชื่อที่อ้างถึง object ชนิด `str`

## 7. Data Types

ชนิดพื้นฐานที่ใช้บ่อย:

```python
age: int = 20
learning_rate: float = 0.001
is_training: bool = True
model_name: str = "tiny-model"
features: list[float] = [1.2, 3.4, 5.6]
metadata: dict[str, str] = {"source": "local"}
```

Type hint ช่วยคนและ static tooling แต่ Python ยังเป็น dynamically typed language

## 8. Control Flow

```python
loss = 0.42

if loss < 0.1:
    status = "excellent"
elif loss < 0.5:
    status = "learning"
else:
    status = "needs work"
```

Loop:

```python
losses = [1.0, 0.8, 0.6]

for epoch, loss in enumerate(losses, start=1):
    print(epoch, loss)
```

## 9. Functions

```python
def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
```

หลักสำคัญ:

- function ควรทำหน้าที่ชัดเจน
- validate input เมื่อ failure ไม่ควรถูกซ่อน
- prefer return value แทน global mutable state

## 10. Classes

```python
class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    def value(self) -> float:
        if self.count == 0:
            raise ValueError("no observations")
        return self.total / self.count
```

ในอนาคต `torch.nn.Module` ก็ใช้แนวคิด object/class แบบนี้

## 11. Iterators and Generators

Generator ช่วย stream data โดยไม่ต้องโหลดทั้งหมดเข้าหน่วยความจำ:

```python
def squares(limit: int):
    for n in range(limit):
        yield n * n
```

แนวคิดนี้จะกลับมาใน DataLoader และ streaming dataset

## 12. Exceptions

อย่าจับ exception กว้างเกินจำเป็น:

```python
try:
    value = float(raw)
except ValueError:
    print("not a number")
```

หลีกเลี่ยง:

```python
try:
    ...
except Exception:
    pass
```

เพราะมันซ่อน root cause

## 13. Modules and Packages

ถ้าไฟล์โต ให้แยกตาม responsibility:

```text
project/
├── app.py
├── data.py
├── metrics.py
└── tests/
```

import:

```python
from pathlib import Path
```

## 14. Linux/Bash

คำสั่งขั้นต่ำ:

```bash
pwd
ls -lah
mkdir -p data/raw
cp source.csv data/raw/
mv old.csv new.csv
find . -type f
grep -R "TODO" .
ps aux
echo "$PATH"
```

อย่ารันคำสั่งที่แก้ระบบด้วย `sudo` ถ้ายังไม่รู้ว่ามันเปลี่ยนอะไร

## 15. Git Workflow

```bash
git status
git switch -c feature/ch01-lab
git diff
git add .
git commit -m "feat: complete chapter 01 lab"
```

คิด Git เป็น graph ของ commits ไม่ใช่ cloud backup อย่างเดียว

## 16. Memory Mental Model

สำหรับพื้นฐาน C/C++:

- **stack** — มักใช้กับ call frames/local data ที่อายุสัมพันธ์กับ scope
- **heap** — dynamic allocation
- **pointer** — ค่า address ที่อ้างตำแหน่งหน่วยความจำ
- **reference** — ชื่อ/handle ที่อ้าง object ตาม semantics ของภาษา

อย่าเหมารวมว่า “Python variable อยู่ stack” หรือ “Python object ทุกอย่างอยู่ heap” เป็นกฎภาษาที่ใช้ reasoning ได้ทุกกรณี; implementation มีรายละเอียดมากกว่านั้น

## 17. Debugging

วงจร:

```text
reproduce
→ read traceback
→ inspect state
→ reduce input
→ form hypothesis
→ test hypothesis
→ fix
→ add regression test
```

## 18. Performance

Big-O ตัวอย่าง:

- index list: โดยทั่วไป O(1)
- scan list: O(n)
- nested scan: มัก O(n²)
- dictionary lookup: average-case โดยทั่วไป O(1)

แต่ performance จริงยังขึ้นกับ memory layout, cache และ constant factor

## 19. Production Perspective

Production code ต้องใส่ใจ:

- input validation
- logging
- configuration
- reproducibility
- tests
- error messages
- dependency isolation

## 20. Research Perspective

Research prototype ที่ reproduce ไม่ได้มีคุณค่าน้อยลงมาก การบันทึก seed, package versions และ command จึงเป็น engineering skill ที่สำคัญต่อ science

## 21. Common Mistakes

1. ใช้ mutable default argument
2. ใช้ global state มากเกิน
3. `except Exception: pass`
4. path แบบ hard-code
5. ใช้ `sudo pip`
6. commit secret
7. แก้ code โดยไม่ reproduce bug
8. ไม่อ่าน traceback
9. สับสน `=` กับ equality
10. สับสน object กับ variable name
11. โหลด dataset ใหญ่ทั้งหมดทั้งที่ stream ได้
12. commit generated files ขนาดใหญ่

## 22. Interview Questions

1. list กับ tuple ต่างกันอย่างไร?
2. mutable/immutable คืออะไร?
3. generator ต่างจาก list อย่างไร?
4. context manager มีไว้ทำอะไร?
5. exception handling ที่ดีควรเป็นแบบใด?
6. process กับ program ต่างกันอย่างไร?
7. environment variable คืออะไร?
8. Git branch คืออะไร?
9. pointer คืออะไร?
10. ทำไม test สำคัญต่อ ML code?

## 23. Chapter Checklist

- [ ] เขียน function พร้อม validation
- [ ] ใช้ class เก็บ state
- [ ] เขียน generator
- [ ] อ่าน traceback
- [ ] ใช้ `pathlib`
- [ ] parse CLI arguments
- [ ] ใช้ Git branch/diff/commit
- [ ] อธิบาย stack/heap/pointer
- [ ] test Dataset Inspector ผ่าน

## 24. What's Next

Chapter 02 จะเปลี่ยน “ตัวเลขในโปรแกรม” ให้เป็นภาษาคณิตศาสตร์ของ AI: vector, matrix, derivative, probability และ statistics
