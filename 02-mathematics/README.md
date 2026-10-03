# Chapter 02 — Mathematics Foundations for AI

## 1. Why This Matters

AI model ส่วนใหญ่คือฟังก์ชันที่รับตัวเลขจำนวนมาก แล้วปรับ parameter เพื่อให้ objective ดีขึ้น ภาษาที่ใช้อธิบายสิ่งนี้คือ:

```text
algebra
→ linear algebra
→ calculus
→ probability
→ statistics
```

ถ้าเห็นสมการแล้วรีบข้าม คุณจะใช้ framework ได้แต่ debug model และอ่าน paper ยาก

## 2. Prerequisites

จาก Chapter 01:

- Python functions
- list/loop
- exceptions
- basic testing

## 3. Learning Objectives

By the end of this chapter you can:

- อ่าน notation เช่น `x ∈ R^n`
- คำนวณ vector operations
- ตรวจ dimension ของ matrix multiplication
- อธิบาย linear transformation
- เข้าใจ determinant/inverse/eigenvalue ในระดับ foundation
- derive derivative ของ polynomial
- ใช้ partial derivative, gradient และ chain rule
- เข้าใจ Jacobian ในฐานะ matrix ของ partial derivatives
- ใช้ probability, conditional probability และ Bayes
- คำนวณ expectation, variance, covariance
- แยก population กับ sample statistics
- อธิบาย hypothesis testing ในระดับพื้นฐาน
- ตรวจสูตรด้วย Python

## 4. Mental Model

```text
data point      → vector
dataset         → matrix
model           → function
model weights   → parameters
error           → loss
direction       → gradient
uncertainty     → probability
evidence        → statistics
```

## 5. Study Order

1. [Algebra & Linear Algebra](theory/01-algebra-linear-algebra.md)
2. [Calculus](theory/02-calculus.md)
3. [Probability & Statistics](theory/03-probability-statistics.md)
4. [Math Toolkit](src/math_toolkit.py)
5. [Exercises](exercises/README.md)
6. [Mini Project](mini-project/README.md)

## 6. Scalars, Vectors, Matrices

Scalar:

```text
x = 3.5
```

Vector:

```text
x = [x₁, x₂, ..., xₙ] ∈ Rⁿ
```

Matrix:

```text
A ∈ R^(m×n)
```

ตัวอย่าง dataset 100 ตัวอย่าง × 4 features:

```text
X ∈ R^(100×4)
```

row = sample, column = feature เป็น convention ที่พบบ่อย

## 7. Dot Product

```text
a · b = Σᵢ aᵢbᵢ
```

ตัวอย่าง:

```text
[1,2,3] · [4,5,6]
= 1×4 + 2×5 + 3×6
= 32
```

ใน linear model:

```text
prediction = w · x + b
```

## 8. Matrix Multiplication

ถ้า:

```text
A ∈ R^(m×n)
B ∈ R^(n×p)
```

แล้ว:

```text
AB ∈ R^(m×p)
```

inner dimensions ต้องเท่ากัน

นี่คือกฎที่ต้องตรวจทันทีเมื่อเจอ neural-network shape error

## 9. Functions

```text
f: X → Y
```

Linear regression ในอนาคต:

```text
f(x) = wx + b
```

Neural network คือ composition ของ functions จำนวนมาก

## 10. Derivatives

Derivative วัด local rate of change:

```text
f'(x) = lim(h→0) [f(x+h)-f(x)]/h
```

ถ้า:

```text
f(x)=x²
```

จะได้:

```text
f'(x)=2x
```

ที่ x=3 slope = 6

## 11. Partial Derivatives & Gradient

ถ้า:

```text
f(x,y)=x²+3y²
```

```text
∂f/∂x = 2x
∂f/∂y = 6y
```

Gradient:

```text
∇f = [2x, 6y]
```

Gradient บอกทิศที่ฟังก์ชันเพิ่มเร็วที่สุดใน local neighborhood; `-∇f` จึงเป็นแนวคิดพื้นฐานของ gradient descent

## 12. Chain Rule

ถ้า:

```text
u=g(x)
y=f(u)
```

แล้ว:

```text
dy/dx = dy/du × du/dx
```

Backpropagation คือการใช้ chain rule ผ่าน computational graph อย่างเป็นระบบ

## 13. Probability

Probability เป็นค่าระหว่าง 0 และ 1

Conditional probability:

```text
P(A|B) = P(A∩B)/P(B)
```

Bayes:

```text
P(A|B) = P(B|A)P(A)/P(B)
```

Chapter Naive Bayes จะใช้ตรง ๆ

## 14. Statistics

Mean:

```text
x̄=(1/n)Σxᵢ
```

Population variance:

```text
σ²=(1/n)Σ(xᵢ-μ)²
```

Sample variance:

```text
s²=(1/(n-1))Σ(xᵢ-x̄)²
```

## 15. Covariance

```text
Cov(X,Y)=E[(X-E[X])(Y-E[Y])]
```

ค่าบวก: มีแนวโน้มเพิ่มไปด้วยกัน  
ค่าลบ: มีแนวโน้มสวนกัน  
ใกล้ศูนย์: linear co-movement ต่ำ แต่ไม่ได้แปลว่า independent

## 16. Numerical Verification

finite difference:

```python
def derivative(f, x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)
```

ใช้ตรวจ analytic derivative ได้ แต่มี numerical error และไม่ใช่ replacement ของ symbolic reasoning

## 17. Determinant / Inverse / Eigen

Foundation intuition:

- determinant บอก scale/orientation effect ของ square linear transformation และช่วยบอก singularity
- inverse คือ transformation ที่ย้อนอีก transformation ได้ เมื่อ inverse มีอยู่
- eigenvector คือ direction ที่ transformation ไม่เปลี่ยนแนว มีเพียง scale
- eigenvalue คือ scale factor ของ eigenvector

PCA ใน Chapter 15 จะกลับมาใช้ eigen/SVD ลึกขึ้น

## 18. Jacobian

ถ้า function รับ vector และคืน vector Jacobian เก็บ partial derivatives:

```text
J_ij = ∂f_i / ∂x_j
```

เป็น generalization สำคัญของ derivative ไปสู่หลายมิติ

## 19. Hypothesis Testing Foundation

องค์ประกอบ:

1. null hypothesis H₀
2. alternative H₁
3. test statistic
4. sampling distribution
5. p-value / decision rule
6. significance level α

**p-value ไม่ใช่ probability ที่ H₀ เป็นจริง**

จะกลับมาเรียนเมื่อทำ experiment/evaluation

## 20. Common Mistakes

1. คูณ matrix โดยไม่ตรวจ shape
2. คิดว่า element-wise multiply = matrix multiply
3. สับสน variance กับ standard deviation
4. ใช้ sample/population variance ผิด
5. คิด correlation = causation
6. คิด covariance 0 = independence เสมอ
7. ลืม domain ของ log
8. ใช้ inverse ทั้งที่ solve system ตรง ๆ เหมาะกว่า
9. ใช้ finite difference ด้วย h เล็กเกินจน round-off error เด่น
10. จำ derivative โดยไม่เข้าใจ chain rule
11. อ่าน sigma notation ไม่ได้
12. ตีความ p-value ผิด

## 21. Interview Questions

1. dot product มีความหมายอย่างไร?
2. ทำไม inner dimensions ต้อง match?
3. gradient คืออะไร?
4. chain rule เกี่ยวกับ backprop อย่างไร?
5. variance ต่างจาก standard deviation อย่างไร?
6. covariance กับ correlation ต่างกันอย่างไร?
7. Bayes theorem ใช้ update belief อย่างไร?
8. eigenvector/eigenvalue คืออะไร?
9. Jacobian คืออะไร?
10. p-value ไม่ได้บอกอะไร?

## 22. Checklist

- [ ] อ่าน `R^(m×n)`
- [ ] dot product ด้วยมือ
- [ ] matrix multiply ด้วยมือ
- [ ] derive quadratic derivative
- [ ] ใช้ chain rule
- [ ] หา gradient ของ function 2 variables
- [ ] คำนวณ Bayes example
- [ ] mean/variance/covariance
- [ ] อธิบาย eigen intuition
- [ ] test math toolkit ผ่าน

## 23. Mini Project

สร้าง **Math Toolkit for Machine Learning** โดยต่อยอด [src/math_toolkit.py](src/math_toolkit.py)

## 24. What's Next

Chapter 03 จะเปลี่ยน math operations เหล่านี้เป็น vectorized computation ด้วย NumPy และใช้ Pandas/Matplotlib จัดการข้อมูลจริง
