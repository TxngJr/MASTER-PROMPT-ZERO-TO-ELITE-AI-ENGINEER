# Chapter 02 Solutions

## Short answers

1. scalar = ค่าเดียว, vector = 1D coordinates, matrix = 2D linear map/table
2. `Σ a_i b_i`
3. local rate of change
4. vector ของ partial derivatives
5. `P(A|B)=P(A∩B)/P(B)`

6. inner dimensions 4 กับ 5 ไม่ match
7. output แต่ละตัวคือ dot(row_i, x)
8. network เป็น composite function จึงต้อง propagate derivatives ผ่าน compositions
9. sample variance แบบ unbiased ใช้ n-1 เมื่อ estimate mean จาก sample
10. association ไม่พิสูจน์ causal mechanism

## Cosine similarity

```python
def cosine_similarity(a, b):
    denominator = l2_norm(a) * l2_norm(b)
    if denominator == 0:
        raise ValueError("zero vector has undefined cosine similarity")
    return dot(a, b) / denominator
```

## Correlation

```text
corr(x,y)=cov(x,y)/(std(x)std(y))
```

ต้อง handle zero standard deviation

## Analytic check

```text
f(x)=x³+2x
f'(x)=3x²+2
```

ที่ x=2 derivative = 14; numerical central difference ควรใกล้ 14

Challenge ควร implement ด้วยตัวเองและเพิ่ม tests แทนการ copy final code
