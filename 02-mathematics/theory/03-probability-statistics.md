# Probability & Statistics

## Random experiment

Probability model เริ่มจาก possible outcomes

ตัวอย่างโยนเหรียญ:

```text
Ω={H,T}
```

Event เป็น subset ของ sample space

## Probability axioms intuition

- `0≤P(A)≤1`
- `P(Ω)=1`
- events ที่ไม่ทับกันรวม probability ด้วยการบวก

Complement:

```text
P(Aᶜ)=1-P(A)
```

## Conditional Probability

```text
P(A|B)=P(A∩B)/P(B)
```

ความหมาย: เมื่อรู้ว่า B เกิดแล้ว probability ของ A เปลี่ยนอย่างไร

## Independence

A และ B independent ถ้า:

```text
P(A∩B)=P(A)P(B)
```

equivalently เมื่อ probabilities เหมาะสม:

```text
P(A|B)=P(A)
```

## Bayes theorem

จาก:

```text
P(A∩B)=P(B|A)P(A)=P(A|B)P(B)
```

จึงได้:

```text
P(A|B)=P(B|A)P(A)/P(B)
```

Interpretation:

```text
posterior ∝ likelihood × prior
```

## Random Variables

Random variable map outcome ไปตัวเลข

Discrete example: จำนวนหัวจากการโยนเหรียญสองครั้ง

## Expected Value

```text
E[X]=Σ_x xP(X=x)
```

Expectation คือ weighted average ระยะยาว ไม่จำเป็นต้องเป็นค่าที่เกิดขึ้นจริงได้

## Variance

```text
Var(X)=E[(X-E[X])²]
```

identity:

```text
Var(X)=E[X²]-E[X]²
```

## Standard Deviation

```text
σ=sqrt(Var(X))
```

หน่วยกลับมาเหมือน X จึงตีความง่ายกว่า variance

## Covariance

```text
Cov(X,Y)=E[(X-E[X])(Y-E[Y])]
```

Correlation standardize covariance:

```text
ρ = Cov(X,Y)/(σ_X σ_Y)
```

Correlation อยู่ใน [-1,1] เมื่อ standard deviations ไม่เป็นศูนย์

## Common Distributions

### Bernoulli
หนึ่ง trial ผล 0/1, parameter p

```text
E[X]=p
Var(X)=p(1-p)
```

### Binomial
จำนวน successes ใน n independent Bernoulli trials

### Gaussian / Normal
กำหนดโดย mean μ และ variance σ²

```text
X ~ N(μ,σ²)
```

### Uniform
ค่าภายในช่วง/ชุดที่กำหนดมี density/probability สม่ำเสมอตาม model

## Population vs Sample

Population = สิ่งทั้งหมดที่เราสนใจ  
Sample = subset ที่สังเกต

Sample mean:

```text
x̄=(1/n)Σxᵢ
```

Unbiased sample variance ใช้ `n-1`:

```text
s²=(1/(n-1))Σ(xᵢ-x̄)²
```

เพราะ mean ถูก estimate จาก sample ทำให้เสียหนึ่ง degree of freedom

## Estimator

Estimator คือ rule/function ที่ใช้ sample ประมาณ parameter

ตัวอย่าง `x̄` estimate population mean μ

คุณสมบัติที่สนใจ:
- bias
- variance
- consistency

## Hypothesis Testing

Example framing:

```text
H₀: μ = 0
H₁: μ ≠ 0
```

test statistic วัดว่าข้อมูล observed ห่างจากสิ่งที่ H₀ คาดแค่ไหน

p-value = probability ภายใต้ H₀ ของผลที่ extreme อย่างน้อยเท่าที่ observed ตามนิยาม test

ไม่ใช่:
- P(H₀ เป็นจริง)
- effect size
- importance ของผล

## Confidence Interval intuition

Interval procedure ที่เมื่อทำ sampling ซ้ำตาม assumptions จะครอบคลุม true parameter ตาม coverage ที่กำหนด

อย่าตีความ frequentist interval แบบตรง ๆ ว่า parameter มี probability 95% อยู่ใน interval หลังเห็นข้อมูลแล้ว

## AI connections

- classifier outputs มักตีความเชิง probability ภายใต้เงื่อนไข/การ calibrate
- Naive Bayes ใช้ Bayes
- losses หลายชนิดมาจาก likelihood
- dataset statistics ใช้ estimate distribution
- evaluation ต้องแยก signal จาก sampling noise
