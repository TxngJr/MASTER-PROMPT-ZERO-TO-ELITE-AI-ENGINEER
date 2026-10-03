# Boosting Mathematics

## AdaBoost exponential-loss intuition

Additive classifier:

```text
F(x)=Σ_t α_t h_t(x)
```

exponential objective:

```text
Σ_i exp(-y_i F(x_i))
```

เมื่อเพิ่ม learner ใหม่:

```text
F_t = F_{t-1} + α h_t
```

เลือก learner ที่ลด weighted exponential loss

sample weights จึง proportional กับ:

```text
exp(-y_i F_{t-1}(x_i))
```

sample ที่ถูกทายผิดมี margin ติดลบและ weight สูงขึ้น

## AdaBoost alpha

ให้ weighted error `ε`

objective สำหรับ weak learner fixed แล้ว minimize ตาม α จะได้:

```text
α = 1/2 ln((1-ε)/ε)
```

เมื่อ ε→0 alpha สูง  
เมื่อ ε→0.5 alpha→0

## Gradient Boosting as Functional Gradient Descent

ปกติ gradient descent optimize parameter vector:

```text
θ ← θ - η∇J(θ)
```

gradient boosting optimize function F:

```text
F_m(x)
=
F_{m-1}(x)
+
η h_m(x)
```

โดย h พยายาม approximate negative functional gradient ของ loss ที่ training points

## Squared loss

```text
L=1/2(y-F)²
```

```text
-∂L/∂F = y-F
```

ดังนั้น fit tree ต่อ residuals

## Logistic-loss intuition

สำหรับ classification pseudo-residual ไม่ใช่ raw label residual อย่างง่ายเสมอ แต่เป็น negative gradient ของ chosen loss ตาม current score/probability

นี่คือเหตุผลที่ mental model ที่ถูกต้องคือ “fit negative gradient” ไม่ใช่ “fit residual” แบบตายตัวทุก objective

## Shrinkage

```text
F_m=F_{m-1}+ηh_m
```

learning rate ต่ำลด contribution ของ tree แต่ละต้น

## Regularization dimensions

Boosting complexity คุมได้หลายแกน:

- number of rounds
- learning rate
- tree depth/leaves
- min samples / min child weight
- row subsampling
- column subsampling
- L1/L2 penalties
- early stopping

ไม่มี parameter เดียวที่แทนทั้งหมด
