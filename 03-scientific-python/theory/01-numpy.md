# NumPy Deep Foundations

## ndarray anatomy

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

print(x.shape)
print(x.ndim)
print(x.dtype)
print(x.size)
print(x.itemsize)
print(x.nbytes)
```

- `shape`: ขนาดแต่ละ axis
- `ndim`: จำนวน axes
- `dtype`: representation ของแต่ละ element
- `size`: จำนวน elements
- `itemsize`: bytes ต่อ element
- `nbytes`: raw element storage โดยประมาณของ array

## Construction

```python
np.zeros((2, 3))
np.ones((2, 3))
np.arange(0, 10, 2)
np.linspace(0.0, 1.0, 5)
```

Random generator แบบ explicit:

```python
rng = np.random.default_rng(42)
x = rng.normal(size=(3, 4))
```

prefer generator object เมื่อ experiment ต้อง reproduce

## Indexing

```python
x[0]
x[0, 1]
x[:, 0]
x[1:, :2]
```

Boolean mask:

```python
mask = x > 0
positive = x[mask]
```

## Vector operations

```python
a + b
a - b
a * b       # element-wise
a / b
a @ b       # matrix multiplication/dot depending dimensions
np.dot(a,b)
```

ระวัง: `*` ไม่ใช่ matrix multiplication

## Broadcasting rules

เทียบ shapes จากขวาไปซ้าย

compatible ถ้า dimension:
- เท่ากัน
- หรือด้านหนึ่งเป็น 1

ตัวอย่าง:

```text
(8, 1, 6, 1)
(   7, 1, 5)
→ (8, 7, 6, 5)
```

อย่าเดา ให้เขียน shape alignment

## Reshape

```python
x.reshape(2, 6)
x.reshape(-1)
```

จำนวน elements ต้องคงเดิม

## Adding/removing axes

```python
x[:, None]
np.expand_dims(x, axis=1)
np.squeeze(x)
```

ใน deep learning การเพิ่ม batch/channel dimension พบเสมอ

## Reductions

```python
x.sum()
x.mean()
x.std()
x.min()
x.max()
```

ด้วย axis:

```python
x.mean(axis=0)
x.mean(axis=1)
```

## Standardization example

ให้ X shape `(n_samples,n_features)`

```python
mean = X.mean(axis=0)
std = X.std(axis=0)

X_standardized = (X - mean) / std
```

shape `(n_features,)` broadcast ไปทุก row

ต้อง handle feature ที่ std=0

## Matrix operations

```python
y = X @ w + b
```

ถ้า:
- X: `(n,d)`
- w: `(d,)`
- b: scalar

ผล y: `(n,)`

นี่คือ vectorized linear model

## Views vs copies

Slicing มักคืน view ในหลายกรณี:

```python
x = np.arange(5)
view = x[1:4]
view[0] = 99
```

x อาจเปลี่ยนตามเพราะ share data

advanced indexing/boolean indexing มี semantics เรื่อง copy ต่างออกไป จึงควรตรวจ `np.shares_memory` เมื่อ correctness สำคัญ

## Numerical dtypes

float32:
- memory ครึ่ง float64
- precision ต่ำกว่า
- เป็น dtype สำคัญใน ML

int types:
- range จำกัด
- overflow semantics ต่างจาก Python int

## Vectorization experiment

```python
import time
import numpy as np

n = 1_000_000
python_values = list(range(n))
numpy_values = np.arange(n)

start = time.perf_counter()
python_result = [x * x for x in python_values]
python_time = time.perf_counter() - start

start = time.perf_counter()
numpy_result = numpy_values * numpy_values
numpy_time = time.perf_counter() - start

print(python_time, numpy_time)
```

ทดลองหลายรอบและอย่าใช้ benchmark นี้เป็นกฎตายตัวสำหรับทุก operation
