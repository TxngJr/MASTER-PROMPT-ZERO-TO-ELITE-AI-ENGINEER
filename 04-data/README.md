# Chapter 04 — Data Fundamentals

## 1. Why This Matters

โมเดลที่ดีไม่สามารถชดเชย dataset ที่ผิดโจทย์ รั่วข้อมูล หรือมี preprocessing ไม่สอดคล้องกันได้

ในงานจริง pipeline มักเป็น:

```text
raw data
   ↓
schema + validation
   ↓
cleaning
   ↓
split
   ↓
fit preprocessing on train
   ↓
transform train/validation/test
   ↓
model
```

ลำดับนี้สำคัญมาก โดยเฉพาะ **split ก่อน fit preprocessing** เพื่อป้องกัน data leakage

## 2. Prerequisites

จาก Batch 01:

- Python functions/classes/tests
- mean, variance, probability
- NumPy arrays
- Pandas DataFrame
- plotting

## 3. Learning Objectives

By the end of this chapter you can:

- แยก sample / feature / target
- ตรวจ schema และ data types
- วิเคราะห์ missing values
- แยก duplicate จริงออกจาก repeated legitimate events
- ตรวจ outliers โดยไม่ลบทิ้งอัตโนมัติ
- ใช้ standardization / min-max scaling
- encode categorical features
- สร้าง features โดยไม่ใช้ข้อมูลอนาคต
- split train / validation / test
- อธิบาย data leakage หลายชนิด
- รับมือ class imbalance ในระดับพื้นฐาน
- สร้าง reusable preprocessing pipeline

## 4. Mental Model

```text
โลกจริง
  ↓ measurement
raw records
  ↓ cleaning
analysis-ready rows
  ↓ feature construction
X, y
  ↓ split
train | validation | test
  ↓ fit(train only)
preprocessor
  ↓ transform
model-ready arrays
```

## 5. Core Vocabulary

- **Sample / observation** — หนึ่งตัวอย่าง
- **Feature** — input variable
- **Target / label** — สิ่งที่ต้องการทำนาย
- **Schema** — ข้อตกลงเรื่อง columns, types, ranges, nullable rules
- **Preprocessing** — การแปลงข้อมูลก่อนเข้า model
- **Feature engineering** — สร้าง feature ใหม่จากข้อมูลเดิม
- **Leakage** — model ได้ข้อมูลที่ตอนใช้งานจริงไม่ควรมี

## 6. Data Types

ประเภทเชิงความหมายสำคัญกว่าชนิด Python อย่างเดียว:

- continuous numeric
- discrete numeric
- binary
- categorical nominal
- categorical ordinal
- text
- image/audio
- datetime/time series
- identifiers

ตัวอย่าง: รหัสไปรษณีย์แม้เป็นตัวเลข ก็ไม่ควรถูกตีความเป็น continuous quantity โดยอัตโนมัติ

## 7. Missing Values

ถามก่อนเติมค่า:

1. missing เพราะอะไร?
2. เกิดใน train และ production แบบเดียวกันหรือไม่?
3. missing เองมี information หรือไม่?
4. การ drop row ทำให้ sample bias หรือไม่?

แนวคิดที่ควรรู้:

- MCAR — Missing Completely At Random
- MAR — Missing At Random เมื่อ conditioned บน observed variables
- MNAR — Missing Not At Random

นี่เป็นกรอบความคิด ไม่ใช่ label ที่พิสูจน์ได้จากตารางเดียวเสมอ

## 8. Numeric Imputation

Simple options:

- mean
- median
- constant
- model-based methods ในบทขั้นสูง

median มักทน outlier กว่า mean

สำคัญที่สุด:

```text
fit imputer on TRAIN
transform TRAIN
transform VALIDATION
transform TEST
```

ห้ามคำนวณ median จาก full dataset ก่อน split

## 9. Categorical Encoding

### One-hot encoding

category:

```text
color ∈ {red, green, blue}
```

แปลงเป็น indicator columns

```text
color_red
color_green
color_blue
```

สำหรับ unseen category ใน production ต้องกำหนด policy เช่น ignore/unknown bucket

### Ordinal encoding

ใช้เมื่อ order มีความหมายจริง เช่น:

```text
low < medium < high
```

อย่าสร้าง order ปลอมให้ nominal categories

## 10. Scaling

### Standardization

```text
z = (x - μ_train) / σ_train
```

mean/std ต้องมาจาก training set

### Min-Max Scaling

```text
x' = (x - min_train)/(max_train-min_train)
```

outlier มีผลมากกว่า standardization

## 11. Outliers

outlier อาจเป็น:

- sensor error
- data entry error
- rare but valid event
- target population จริง

อย่าทำ:

```text
outlier → delete
```

โดยอัตโนมัติ

ให้ตรวจ context, plot และ data provenance ก่อน

## 12. Duplicates

`df.duplicated()` หา row ซ้ำเชิงค่า แต่ไม่ได้ตอบว่าซ้ำ “เชิงความหมาย” หรือไม่

เช่น purchase สองรายการที่เหมือนกันทุก column อาจเป็น transaction คนละเหตุการณ์

ต้องรู้ key เช่น:

```text
event_id
customer_id + timestamp + sequence
```

## 13. Feature Engineering

ตัวอย่าง:

- age จาก birth_date
- total_price = quantity × unit_price
- hour/day-of-week จาก timestamp
- log transform สำหรับ positive skew
- interaction terms

กฎสำคัญ:

> ทุก feature ต้องสร้างได้ด้วยข้อมูลที่มี ณ เวลาทำนายจริง

ถ้าใช้ข้อมูลอนาคต = leakage

## 14. Dataset Splitting

ทั่วไป:

- training — fit parameters
- validation — เลือก hyperparameters/model
- test — final unbiased estimate หลังตัดสินใจเสร็จ

ตัวอย่าง:

```text
70% train
15% validation
15% test
```

ไม่มีกฎว่าต้องเป็นสัดส่วนนี้เสมอ

## 15. Random Split vs Time Split

Random split เหมาะเมื่อ samples approximately exchangeable

Time-dependent data:

```text
past → train
later → validation
future → test
```

ห้าม shuffle future เข้าอดีตถ้า deployment จริงทำนายอนาคต

## 16. Stratification

Classification dataset ที่ class imbalance มาก อาจใช้ stratified split เพื่อรักษาสัดส่วน classes ใกล้เคียงกัน

แต่ stratification ไม่แก้ปัญหา imbalance ทั้งหมด

## 17. Data Leakage

### Target leakage

feature มีข้อมูลที่เกิดหลัง target หรือ encode target โดยตรง

### Train-test contamination

fit preprocessing กับ full dataset

### Duplicate leakage

record/บุคคล/เหตุการณ์เดียวกันโผล่ทั้ง train และ test

### Group leakage

ตัวอย่างจาก entity เดียวกันกระจายหลาย split ทั้งที่ deployment เจอ entity ใหม่

### Time leakage

ใช้ future information ทำนายอดีต

## 18. Imbalanced Data

Accuracy อาจหลอกได้

ถ้า 99% เป็น class 0 การทาย 0 ตลอดได้ accuracy 99% แต่ model ไม่มีประโยชน์ต่อ minority class

บท classification จะเรียน precision/recall/F1/ROC/PR ลึกขึ้น

## 19. Data Contracts

กำหนดอย่างน้อย:

- column names
- required/optional
- types
- ranges
- allowed categories
- uniqueness
- null policy

ตัวอย่าง:

```text
age: float, nullable, 0..120
country: string, required
customer_id: string, unique
```

## 20. From Scratch

ดู [src/preprocessing.py](src/preprocessing.py)

เราสร้าง:

- train/validation/test index split
- Standardizer ที่มี `fit` / `transform`
- median imputer
- one-hot encoder แบบพื้นฐาน

เป้าหมายคือเข้าใจ state ที่ preprocessing เรียนจาก train

## 21. scikit-learn Perspective

ใน production-oriented workflow นิยมประกอบ preprocessing ด้วย:

- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`
- `Pipeline`

ข้อดีคือ fit/transform state ถูกผูกกับ pipeline ลดโอกาส leakage

## 22. Debugging Checklist

ก่อน train model:

```python
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df.duplicated().sum())
print(df.head())
```

หลัง preprocess:

- shape ตรงคาดไหม?
- NaN เหลือไหม?
- category ใหม่ handle อย่างไร?
- scaler fit จาก train เท่านั้นไหม?
- target หลุดเข้า X หรือไม่?

## 23. Common Mistakes

1. impute ก่อน split
2. scale ก่อน split
3. encode target เป็น feature
4. ใช้ ID เป็น continuous feature
5. drop outlier ทุกตัว
6. drop missing ทุก row
7. random split time series
8. duplicate entity ข้าม splits
9. one-hot แล้ว production เจอ category ใหม่จนพัง
10. fit preprocessing แยกจาก model แล้วลืมใช้ตัวเดิมตอน inference
11. feature จากอนาคต
12. evaluate ซ้ำบน test จน test กลายเป็น validation

## 24. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] อธิบาย sample/feature/target
- [ ] ทำ schema audit
- [ ] วิเคราะห์ missing/outliers/duplicates
- [ ] split ก่อน fit transforms
- [ ] implement standardizer
- [ ] explain leakage 5 แบบ
- [ ] encode categorical data
- [ ] อธิบาย time split
- [ ] test preprocessing ผ่าน

## 26. What's Next

Chapter 05 จะนิยามว่า “Machine Learning” กำลัง optimize อะไร วัดอย่างไร และทำไม model ที่ train loss ต่ำยังอาจใช้งานจริงได้แย่
