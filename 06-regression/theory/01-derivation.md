# Linear Regression Derivation

## Scalar case

```text
ŷᵢ = wxᵢ+b
eᵢ = ŷᵢ-yᵢ
J = (1/n)Σeᵢ²
```

สำหรับ w:

```text
∂J/∂w
= (1/n)Σ ∂(eᵢ²)/∂w
= (1/n)Σ 2eᵢ ∂eᵢ/∂w
= (2/n)Σ eᵢxᵢ
```

สำหรับ b:

```text
∂J/∂b
= (2/n)Σeᵢ
```

## Vectorized case

```text
e = Xw + b1 - y
J = (1/n)eᵀe
```

gradient:

```text
∇w J = (2/n)Xᵀe
```

intercept:

```text
∂J/∂b = (2/n)1ᵀe
```

## Normal equations

Augment:

```text
X̃=[1 X]
θ=[b;w]
```

Objective:

```text
J(θ)=||X̃θ-y||²
```

Gradient:

```text
∇θJ=2X̃ᵀ(X̃θ-y)
```

set to zero:

```text
X̃ᵀX̃θ=X̃ᵀy
```

ถ้า invertible:

```text
θ=(X̃ᵀX̃)⁻¹X̃ᵀy
```

แต่ numerical implementation ใช้ least-squares/SVD/QR-based routines มากกว่า explicit inverse
