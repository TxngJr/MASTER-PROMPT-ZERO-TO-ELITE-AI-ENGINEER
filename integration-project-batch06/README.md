# Batch 06 Integration Project — Anomaly-Aware Forecasting & Neural Baseline Lab

รวม Chapters 16–18:

~~~text
time series
   ↓
chronological split
   ↓
causal anomaly features
   ↓
IsolationForest fit on TRAIN only
   ↓
lag-window supervised dataset
   ↓
Seasonal Naive
Linear Autoregression
MLPRegressor
   ↓
validation RMSE
   ↓
model selection
   ↓
final future test
~~~

## Goal

แยก 3 งานให้ชัด:

1. **Detection** — จุดไหนดูผิดปกติ?
2. **Forecasting** — ค่าถัดไปควรเป็นเท่าไร?
3. **Neural Representation** — MLP ใช้ lag window อย่างไร?

Anomaly detector ไม่ได้ลบข้อมูลโดยอัตโนมัติ; project เพียงรายงาน scores/flags และวัดว่ามี anomalies กระจายตรงไหน

## Run

~~~bash
source .venv/bin/activate
python -m pip install -r requirements-batch06.txt

python integration-project-batch06/src/forecasting_neural_lab.py   --output-dir reports/batch06   --seed 42
~~~

## Models

- seasonal naive baseline
- linear autoregression on lag features
- MLPRegressor on lag features

## Leakage Rules

- chronological split only
- anomaly detector fit on training timeline only
- lag row at time t uses values before t only
- StandardScaler lives inside the MLP Pipeline
- validation selects model
- final test is evaluated once

## Required Extensions

1. add statsmodels ARIMA candidate
2. optional Prophet candidate
3. rolling-origin evaluation
4. multi-step recursive forecast
5. contextual anomaly detector using hour/day features
6. compare anomaly-removal vs robust training carefully
7. add prediction intervals
8. MLP hidden-layer sweep
9. anomaly-aware loss weighting
10. regime-shift scenario

## Mastery Questions

- ทำไม anomaly detector fit full timeline จึง leak?
- ทำไม anomaly ไม่ควรถูกลบอัตโนมัติ?
- one-step lag evaluation ต่างจาก recursive multi-step อย่างไร?
- MLP training ใช้ backprop แม้ Chapter 18 ยังไม่ derive อย่างไร?
- validation/test chronology สำคัญอย่างไร?
