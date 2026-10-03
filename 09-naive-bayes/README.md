# Chapter 09 — Naive Bayes

## 1. Why This Matters

Naive Bayes เป็น classifier ที่เรียบง่ายแต่สอนแนวคิดสำคัญมาก:

- Bayes theorem
- generative classification
- likelihood
- prior
- conditional independence
- maximum likelihood
- smoothing
- log-space numerical stability

```text
class prior P(y)
      +
feature likelihoods P(x_j|y)
      ↓
posterior score
      ↓
argmax class
```

## 2. Bayes Theorem

```text
P(y|x) = P(x|y)P(y) / P(x)
```

สำหรับ classification เราเปรียบเทียบ classes และ `P(x)` เหมือนกันทุก class:

```text
P(y|x) ∝ P(x|y)P(y)
```

จึงเลือก:

```text
ŷ = argmax_y P(x|y)P(y)
```

## 3. Naive Assumption

ถ้า features `x₁,...,x_d` independent conditional on class:

```text
P(x|y) = Π_j P(x_j|y)
```

จึงได้:

```text
P(y|x) ∝ P(y) Π_j P(x_j|y)
```

ในโลกจริง assumption นี้มักไม่จริงเต็มที่ แต่ model ยังทำงานดีในบาง domain โดยเฉพาะ text/count features

## 4. Why "Generative"?

Naive Bayes model joint distribution ผ่าน:

```text
P(y)P(x|y)
```

ต่างจาก Logistic Regression ที่ model `P(y|x)` โดยตรง

นี่คือ conceptual distinction ระหว่าง generative และ discriminative classifiers

## 5. Log Space

product ของ probabilities เล็กจำนวนมาก:

```text
0.001 × 0.002 × ...
```

อาจ underflow

ใช้ log:

```text
log P(y|x)
∝
log P(y) + Σ_j log P(x_j|y)
```

product กลายเป็น sum และ numerical stability ดีขึ้น

## 6. Gaussian Naive Bayes

ใช้กับ continuous numeric features โดยสมมติ:

```text
x_j | y=c ~ Normal(μ_cj, σ²_cj)
```

density:

```text
P(x_j|y=c)
=
1/sqrt(2πσ²)
×
exp(-(x_j-μ)²/(2σ²))
```

log density:

```text
-0.5 log(2πσ²)
-
(x-μ)²/(2σ²)
```

training estimate ต่อ class/feature:

- mean
- variance
- class prior

## 7. Gaussian Variance Smoothing

ถ้า variance = 0:

```text
division by zero
log(0)
```

implementation จึงเพิ่ม epsilon/smoothing ให้ variance

production libraries มี strategy ที่ละเอียดกว่า

## 8. Multinomial Naive Bayes

เหมาะกับ non-negative counts เช่น bag-of-words

ให้:

```text
N_cj = count ของ feature j ใน class c
N_c = Σ_j N_cj
```

smoothed estimate:

```text
θ_cj =
(N_cj + α) /
(N_c + αd)
```

เมื่อ `α=1` เรียก Laplace smoothing

## 9. Why Smoothing?

ถ้า word ไม่เคยพบใน class:

```text
P(word|class)=0
```

เพราะ Naive Bayes คูณ likelihoods ค่า zero หนึ่งตัวทำ posterior product เป็น zero

smoothing ป้องกันปัญหานี้

## 10. Bernoulli Naive Bayes

เหมาะกับ binary features:

```text
word present = 1
word absent = 0
```

มัน model ทั้ง presence และ absence

ต่างจาก MultinomialNB ที่สนใจ counts/frequencies

## 11. Class Priors

```text
P(y=c)=n_c/n
```

base rate ของ class มีผลต่อ prediction

ถ้า classes imbalance มาก prior จะสะท้อน training distribution

ถ้า deployment prevalence เปลี่ยน probability interpretation อาจ shift

## 12. Maximum Likelihood

GaussianNB:
- estimate class mean/variance จาก samples ใน class

MultinomialNB:
- estimate token probabilities จาก counts

นี่คือ parameter estimation จาก likelihood

## 13. From Scratch

[src/naive_bayes.py](src/naive_bayes.py) มี:

- GaussianNBFromScratch
- MultinomialNBFromScratch

ทั้งคู่ทำ computation ใน log space

## 14. GaussianNB Steps

Training:

```text
for each class:
  prior
  feature means
  feature variances
```

Prediction:

```text
log prior
+
Σ Gaussian log likelihood
→ choose largest score
```

## 15. MultinomialNB Steps

Training:

```text
class counts
feature counts per class
+ alpha smoothing
normalize
take logs
```

Prediction:

```text
log prior + X @ log(feature probabilities)
```

## 16. Text Classification Connection

Classic text pipeline:

```text
documents
   ↓
tokenization
   ↓
count matrix
   ↓
Multinomial Naive Bayes
```

ต่อไป NLP chapters จะทำ representation ที่ซับซ้อนขึ้น แต่แนวคิด probabilistic language evidence ยังสำคัญ

## 17. Feature Dependence Problem

ตัวอย่าง text:

```text
"machine"
"learning"
```

features สองคำไม่ independent จริง

Naive Bayes อาจ double-count correlated evidence

นี่เป็นข้อจำกัดหลัก

## 18. Gaussian Assumption Problem

ถ้า feature distribution ใน class:

- skew มาก
- multimodal
- bounded แปลก
- heavy-tailed

Gaussian likelihood อาจไม่เหมาะ

transform feature หรือ model อื่นอาจดีกว่า

## 19. Probability Interpretation

Naive Bayes posterior estimates อาจ overconfident เพราะ independence assumption

ดังนั้น high `predict_proba` ไม่เท่ากับ calibrated confidence เสมอ

## 20. Complexity

GaussianNB:

training:
```text
O(nd)
```

prediction:
```text
O(n_query × classes × d)
```

MultinomialNB efficient มากกับ sparse text counts

## 21. Comparison With Logistic Regression

Logistic:
- discriminative
- learns decision boundary directly
- iterative optimization
- often benefits from more data

Naive Bayes:
- generative
- simple parameter estimates
- fast training
- strong baseline on text/count data

ไม่มี model ไหนชนะเสมอ

## 22. Comparison With KNN

KNN:
- local geometry
- stores examples
- expensive inference

Naive Bayes:
- compact learned statistics
- fast inference
- strong distribution assumptions

## 23. Common Mistakes

1. ใช้ MultinomialNB กับ negative feature values
2. คูณ probabilities ตรง ๆ จน underflow
3. ไม่มี smoothing
4. คิด independence assumption ว่าเป็นจริงเพราะชื่อ model
5. ใช้ GaussianNB โดยไม่ inspect feature distributions
6. ตีความ probability เป็น calibrated confidence โดยอัตโนมัติ
7. target leakage
8. fit vocabulary/full text preprocessing ก่อน split
9. compare models คนละ split
10. ignore prior shift

## 24. scikit-learn

variants สำคัญ:

- `GaussianNB`
- `MultinomialNB`
- `BernoulliNB`

scikit-learn documentation ระบุ Gaussian likelihood และ Multinomial smoothing equations ตรงกับ derivation ในบทนี้

## 25. Exercises & Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 26. Checklist

- [ ] Bayes theorem
- [ ] prior/likelihood/posterior
- [ ] conditional independence
- [ ] log-space
- [ ] GaussianNB
- [ ] MultinomialNB
- [ ] Laplace smoothing
- [ ] BernoulliNB concept
- [ ] limitations
- [ ] sklearn comparison

## 27. What's Next

Batch ถัดไปจะเข้าสู่ Decision Trees, Random Forest และ Gradient Boosting ซึ่งเปลี่ยนจาก linear/probabilistic/local models ไปสู่ tree-based models และ ensembles
