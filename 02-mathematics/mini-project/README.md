# Mini Project — Math Toolkit for Machine Learning

Starter: [../src/math_toolkit.py](../src/math_toolkit.py)

## Required additions

เพิ่ม:

- scalar-vector multiplication
- matrix addition
- cosine similarity
- correlation
- 2×2 determinant
- numerical gradient สำหรับ `f: R²→R`

## Verification

ทุก operation ต้องมี:

1. example ที่คำนวณด้วยมือได้
2. unit test
3. invalid-shape/input test
4. docstring ระบุ dimensions

## Final experiment

ให้ function:

```text
f(x,y)=(x-3)²+2(y+1)²
```

1. derive analytic gradient
2. implement numerical gradient
3. เปรียบเทียบที่อย่างน้อย 5 จุด
4. อธิบายว่าทำไม numerical กับ analytic ไม่เท่ากันเป๊ะ
