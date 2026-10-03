# Math Reference

ไฟล์นี้เป็น cheat sheet ไม่แทน Chapter 02

## Mean
```text
x̄ = (1/n) Σ x_i
```

## Population variance
```text
Var(X) = (1/n) Σ (x_i - x̄)^2
```

## Dot product
```text
a · b = Σ a_i b_i
```

## Matrix-vector product
ถ้า A ∈ R^(m×n), x ∈ R^n:
```text
y = Ax
y_i = Σ_j A_ij x_j
```

## Derivative
```text
f'(x) = lim(h→0) [f(x+h)-f(x)]/h
```

## Chain rule
```text
d/dx f(g(x)) = f'(g(x)) g'(x)
```

## Gradient
```text
∇f = [∂f/∂x_1, ..., ∂f/∂x_n]
```

## Probability
```text
P(A|B) = P(A∩B)/P(B)
P(A|B) = P(B|A)P(A)/P(B)
```

## Numerical habit
derive ด้วยมือก่อน แล้วตรวจด้วยตัวเลขเล็ก ๆ ใน Python
