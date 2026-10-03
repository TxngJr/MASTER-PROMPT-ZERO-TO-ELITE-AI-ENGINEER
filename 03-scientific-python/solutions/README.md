# Chapter 03 Solutions

## Core answers

1. shape=ขนาด axes, ndim=จำนวน axes, dtype=representation ของ elements
2. `*` element-wise; `@` matrix product semantics
3. rule สำหรับขยาย dimensions ที่ compatible โดยไม่ materialize copies แบบ naive
4. labeled 2D tabular structure
5. distribution ของ numeric variable

6. `(100,4)(4,3)→(100,3)`
7. axis 0 ถูก reduce ออก เหลือ feature axis
8. loop หลักอยู่ใน compiled optimized code และใช้ contiguous numeric memory ได้ดี
9. missing mechanism อาจมี information/bias
10. association ไม่ยืนยัน causal direction/mechanism

## Standardization pattern

```python
mean = X.mean(axis=0)
std = X.std(axis=0)
safe_std = np.where(std == 0, 1.0, std)
Z = (X - mean) / safe_std
```

## Groupby

```python
df.groupby("group")["score"].mean()
```

## Scatter

```python
fig, ax = plt.subplots()
ax.scatter(df["study_hours"], df["score"])
ax.set_xlabel("Study hours")
ax.set_ylabel("Score")
fig.tight_layout()
```

Challenge ควรเก็บ code, results และข้อจำกัดของ benchmark ไว้ใน mini-project report
