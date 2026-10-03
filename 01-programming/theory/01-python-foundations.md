# Python Foundations — From Values to Reusable Programs

## Values, names, objects

ดู:

```python
x = [1, 2, 3]
y = x
y.append(4)
print(x)
```

ผลคือ `[1, 2, 3, 4]` เพราะ `x` และ `y` อ้าง list object เดียวกัน

ถ้าต้องการ object ใหม่:

```python
y = x.copy()
```

นี่เป็นพื้นฐานสำคัญมาก เพราะ bug ใน data pipeline มักเกิดจากการแก้ object ร่วมโดยไม่ตั้งใจ

## Equality vs identity

```python
a = [1, 2]
b = [1, 2]

print(a == b)  # values equal
print(a is b)  # different objects
```

ใช้ `is` กับ identity เช่น `value is None` ไม่ใช้แทน `==`

## Mutability

Mutable:
- list
- dict
- set

Immutable:
- int
- float
- bool
- str
- tuple (ตัว tuple เอง)

### Mutable default pitfall

ผิด:

```python
def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket
```

default object ถูกสร้างครั้งเดียว

ถูก:

```python
def add_item(item: int, bucket: list[int] | None = None) -> list[int]:
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
```

## Functions as abstraction

Function ที่ดีควรมี contract:

```python
def normalize_0_1(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    return (value - minimum) / (maximum - minimum)
```

Contract:

- input เป็นตัวเลข
- maximum > minimum
- output เป็น normalized scalar

## Scope

Python หา name โดยแนวคิด LEGB:

1. Local
2. Enclosing
3. Global
4. Built-in

หลีกเลี่ยงการพึ่ง global state ใน core calculations

## Comprehensions

```python
values = [1, 2, 3, 4]
squares = [x * x for x in values]
even = [x for x in values if x % 2 == 0]
```

ถ้า expression ซับซ้อนเกิน อ่านยาก ให้กลับไปใช้ loop

## Iteration protocol

`for` ต้องการ iterable

```python
for item in [10, 20, 30]:
    print(item)
```

ภายใต้แนวคิด:

```python
iterator = iter([10, 20, 30])
print(next(iterator))
```

Generator ใช้ `yield` สร้าง iterator แบบ lazy

## File handling

ใช้ context manager:

```python
from pathlib import Path

path = Path("example.txt")

with path.open("w", encoding="utf-8") as f:
    f.write("hello\n")
```

เมื่อออกจาก `with` file จะถูก close

## Dataclasses

สำหรับ record ที่มี fields ชัด:

```python
from dataclasses import dataclass

@dataclass
class Experiment:
    name: str
    seed: int
    learning_rate: float
```

## Type hints

```python
def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return sum(x * y for x, y in zip(a, b))
```

Type hints ไม่ได้ replace runtime validation

## Decorators — mental model

Decorator รับ function แล้วคืน function:

```python
from collections.abc import Callable

def trace(fn: Callable[[], None]) -> Callable[[], None]:
    def wrapper() -> None:
        print("start")
        fn()
        print("end")
    return wrapper
```

อย่ารีบใช้ decorator ซับซ้อนถ้าฟังก์ชันธรรมดาอ่านง่ายกว่า

## Context managers

แนวคิด:

```text
acquire resource
try:
    use resource
finally:
    release resource
```

`with` ช่วย encode pattern นี้

## Functional habits useful for AI

- pure calculation functions ทดสอบง่าย
- explicit inputs/outputs
- deterministic seed เมื่อ experiment ต้อง reproduce
- อย่าปน I/O, plotting และ core math ใน function เดียวถ้าไม่จำเป็น
