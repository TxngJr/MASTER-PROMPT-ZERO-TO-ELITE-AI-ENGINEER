# Matplotlib & Jupyter

## Why visualize?

summary statistic เดียวซ่อน distribution ได้ Plot ช่วยเห็น:

- skew
- multimodality
- outliers
- nonlinear relationships
- time trends
- group differences

## Object-oriented Matplotlib

prefer:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [2, 4, 8])
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.set_title("Example")
fig.tight_layout()
plt.show()
```

`fig` = whole figure  
`ax` = plotting area

## Histogram

ตอบ distribution ของตัวแปรเดียว:

```python
fig, ax = plt.subplots()
ax.hist(values, bins=20)
```

bin choice เปลี่ยนสิ่งที่มองเห็น จึงควรลองหลาย resolution

## Scatter

ตอบ relationship ของสอง numeric variables:

```python
ax.scatter(df["hours"], df["score"])
```

ดู:
- direction
- nonlinear pattern
- clusters
- outliers
- changing variance

## Line plot

เหมาะเมื่อ x มี order โดยเฉพาะ time:

```python
ax.plot(time, value)
```

อย่าใช้ line เชื่อม categories ที่ไม่มี continuity โดยไม่ตั้งใจ

## Save reproducibly

```python
fig.savefig("report.png", dpi=150, bbox_inches="tight")
```

ใน automated pipeline ควร save มากกว่า rely on GUI

## Jupyter execution model

Notebook kernel มี state

ถ้ารัน:

```text
cell 7
cell 2
cell 10
```

อาจได้ state ที่คนอื่น reproduce ไม่ได้

ก่อน commit notebook:
1. Restart kernel
2. Run all
3. ตรวจ errors
4. เคลียร์ข้อมูลลับ/outputs ใหญ่
5. ย้าย reusable logic ไป modules

## Exploration vs production

Notebook:
- ask questions
- inspect
- visualize
- prototype

Python module:
- reusable transforms
- validated logic
- tests
- CLI/API

ทั้งสองใช้ร่วมกัน ไม่ใช่เลือกอย่างใดอย่างหนึ่ง
