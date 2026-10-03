# Algebra & Linear Algebra

## Numbers and Variables

Variable ในคณิตศาสตร์แทนค่าที่อาจเปลี่ยนได้

```text
y = 2x + 3
```

เมื่อ x=4:

```text
y=11
```

## Exponents

```text
a^m a^n = a^(m+n)
(a^m)^n = a^(mn)
a^-n = 1/a^n
```

Exponential growth และ powers พบใน scaling laws, probabilities และ optimization

## Logarithms

```text
log_b(x)=y ⇔ b^y=x
```

สำคัญ:

```text
log(ab)=log a + log b
log(a^k)=k log a
```

ใช้ log เปลี่ยน product ของ probabilities ให้เป็น sum ช่วย numerical stability และ optimization

## Functions

Function mapping input ไป output:

```text
f(x)=x²
```

composition:

```text
(f∘g)(x)=f(g(x))
```

Neural network คือ composition หลาย layers

## Vectors

```text
x=[x₁,x₂,...,xₙ]^T
```

Operations:

```text
x+y
cx
x·y
||x||
```

Euclidean norm:

```text
||x||₂ = sqrt(Σ xᵢ²)
```

Distance:

```text
d(x,y)=||x-y||₂
```

KNN จะใช้แนวคิดนี้

## Dot product as similarity

```text
x·y = ||x|| ||y|| cos θ
```

ดังนั้นถ้า normalize vector แล้ว dot product เชื่อมกับ cosine similarity ซึ่งจะกลับมาใน embeddings/RAG

## Matrices

```text
A =
[a11 a12
 a21 a22]
```

Matrix สามารถมองเป็น:
- ตารางตัวเลข
- linear map
- collection ของ row/column vectors

## Matrix-vector multiplication

```text
y=Ax
```

แต่ละ output เป็น dot product ระหว่าง row ของ A กับ x

ตัวอย่าง:

```text
A = [1 2
     3 4]

x = [5
     6]

Ax = [17
      39]
```

## Matrix multiplication

```text
C=AB
C_ij = Σ_k A_ik B_kj
```

shape rule:

```text
(m×n)(n×p) → (m×p)
```

## Transpose

```text
(A^T)_ij=A_ji
```

shape `m×n → n×m`

## Identity

```text
AI=IA=A
```

## Inverse

ถ้า A invertible:

```text
A⁻¹A=I
```

สำหรับระบบ `Ax=b` ทางทฤษฎีเขียน `x=A⁻¹b` ได้ แต่ numerical computing มัก prefer solve algorithm โดยตรงแทนสร้าง inverse

## Determinant

2×2:

```text
det([a b; c d]) = ad-bc
```

ถ้า determinant = 0 matrix singular ไม่มี inverse

Geometric view: absolute determinant คือ factor ที่ linear transformation scale volume

## Linear Independence

vectors `v₁,...,v_k` independent ถ้า:

```text
c₁v₁+...+c_kv_k=0
```

เกิดได้เฉพาะเมื่อ coefficients ทุกตัวเป็น 0

## Basis and Rank

Basis คือ independent vectors ที่ span space

Rank บอกจำนวน independent directions ที่ matrix represent ได้

Low-rank idea จะกลับมาใน PCA, matrix factorization และ LoRA

## Eigenvectors and Eigenvalues

```text
Av=λv
```

v คือ eigenvector, λ คือ eigenvalue

A transform v แล้ว direction เดิมแต่ scale ด้วย λ

## Connection to AI

- linear layer: matrix multiplication
- embeddings: vectors
- attention: dot products + matrix products
- PCA: eigen/SVD
- LoRA: low-rank matrices
- batching: เพิ่ม dimension ให้ computation
