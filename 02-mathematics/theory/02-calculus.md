# Calculus for Optimization

## Why calculus?

Training model คือการถามว่า:

> ถ้าเปลี่ยน parameter นิดเดียว loss จะเปลี่ยนอย่างไร?

Derivative คือเครื่องมือวัด sensitivity นี้

## Limit intuition

Derivative:

```text
f'(x)=lim(h→0) [f(x+h)-f(x)]/h
```

ส่วน quotient คือ slope ระหว่างสองจุด และ limit ทำให้สองจุดเข้าใกล้กัน

## Derive x²

```text
f(x)=x²

f(x+h)=(x+h)²
       =x²+2xh+h²

[f(x+h)-f(x)]/h
=[2xh+h²]/h
=2x+h

lim(h→0)(2x+h)=2x
```

ดังนั้น:

```text
d(x²)/dx=2x
```

## Rules

```text
d(c)/dx = 0
d(x^n)/dx = n x^(n-1)
d(f+g)/dx = f' + g'
d(fg)/dx = f'g + fg'
```

Quotient:

```text
d(f/g)/dx = (f'g-fg')/g²
```

## Chain Rule

ถ้า:

```text
y=f(u)
u=g(x)
```

then:

```text
dy/dx = dy/du × du/dx
```

ตัวอย่าง:

```text
y=(3x+1)²
u=3x+1

dy/du=2u
du/dx=3

dy/dx=6(3x+1)
```

## Partial Derivatives

สำหรับ:

```text
f(x,y)=x²+xy+y²
```

ถือ y คงที่เมื่อหา partial ตาม x:

```text
∂f/∂x=2x+y
```

และ:

```text
∂f/∂y=x+2y
```

## Gradient

```text
∇f=[
 ∂f/∂x₁
 ...
 ∂f/∂xₙ
]
```

ตัวอย่าง f(x,y)=x²+y²:

```text
∇f=[2x,2y]
```

ที่ (3,4):

```text
∇f=[6,8]
```

## Directional intuition

Gradient ชี้ทิศ steepest local increase ภายใต้ Euclidean geometry

ดังนั้น minimization แบบพื้นฐานเดิน:

```text
θ_new = θ_old - η ∇L(θ)
```

Chapter 05 จะศึกษากฎนี้ในบริบท ML

## Jacobian

ให้:

```text
f: R^n → R^m
```

Jacobian:

```text
J =
[∂f1/∂x1 ... ∂f1/∂xn
 ...
 ∂fm/∂x1 ... ∂fm/∂xn]
```

gradient เป็นกรณีพิเศษเมื่อ output เป็น scalar

## Numerical differentiation

Central difference:

```text
f'(x) ≈ [f(x+h)-f(x-h)]/(2h)
```

Trade-off:
- h ใหญ่เกิน → approximation error
- h เล็กเกิน → floating-point round-off/cancellation

## Computational graphs

```text
x ──► multiply ──► u ──► square ──► y
      ×3
```

Backward calculation ใช้ local derivatives และ chain rule นี่คือ mental model ของ autograd

## Exercises in thought

ถ้า:
```text
L=(wx+b-y)²
```

ถาม:
- prediction คือ node ไหน?
- error คือ node ไหน?
- `∂L/∂w` ต้องผ่าน chain rule กี่ช่วง?

คำถามนี้จะกลับมาเต็มรูปใน Backpropagation
