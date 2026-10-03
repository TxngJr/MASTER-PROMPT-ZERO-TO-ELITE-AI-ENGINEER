# Pandas for Tabular Data

## Series and DataFrame

Series = labeled 1D values  
DataFrame = labeled 2D table

```python
import pandas as pd

df = pd.DataFrame(
    {
        "name": ["A", "B"],
        "score": [80.0, 90.0],
    }
)
```

## Load

```python
df = pd.read_csv("data.csv")
```

ทันทีหลัง load:

```python
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
```

## Types

```python
print(df.dtypes)
```

อย่าคิดว่า column ที่หน้าตาเหมือนตัวเลขถูก parse เป็น numeric เสมอ

Convert:

```python
df["score"] = pd.to_numeric(df["score"], errors="coerce")
```

`errors="coerce"` เปลี่ยน invalid values เป็น NaN — ต้องตรวจจำนวนที่เสียไป

## Missing data

```python
df.isna().sum()
df.dropna()
df.fillna({"score": df["score"].median()})
```

แต่ operation ที่ถูกต้องขึ้นกับ data-generating process ไม่ใช่เลือกตามความสะดวก

## Duplicates

```python
df.duplicated().sum()
df.drop_duplicates()
```

ถามว่า duplicate หมายถึง:
- duplicate จริง?
- repeated legitimate event?
- multiple measurements?

## Selection

```python
df["score"]
df[["name", "score"]]
df.loc[df["score"] > 80, ["name", "score"]]
df.iloc[:5, :2]
```

## Assignment

prefer:

```python
mask = df["score"] < 0
df.loc[mask, "score"] = pd.NA
```

หลีกเลี่ยง chained assignment ที่ semantics ไม่ชัด

## Groupby

```python
summary = (
    df.groupby("group", dropna=False)["score"]
      .agg(["count", "mean", "median", "std"])
)
```

Mental model:

```text
split
→ apply
→ combine
```

## Merge

```python
orders.merge(customers, on="customer_id", how="left")
```

ก่อน merge ตรวจ:
- key uniqueness
- expected row counts
- one-to-one / one-to-many / many-to-many

many-to-many merge โดยไม่ตั้งใจทำให้จำนวน rows ระเบิดได้

## Sorting

```python
df.sort_values("score", ascending=False)
```

## Aggregation

```python
df["score"].agg(["count", "mean", "std", "min", "max"])
```

## Correlation

```python
numeric = df.select_dtypes(include="number")
corr = numeric.corr()
```

Correlation เป็น descriptive relationship ไม่ใช่ causal proof

## Tidy pipeline

แทนการ overwrite raw data:

```text
raw.csv
  ↓
load
  ↓
validate
  ↓
clean
  ↓
analysis-ready dataframe
  ↓
summary/plots
```

เก็บ raw immutable เมื่อทำ project จริง
