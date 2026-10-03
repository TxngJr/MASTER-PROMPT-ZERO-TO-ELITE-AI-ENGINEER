# Chapter 17 — Time Series Forecasting

## 1. Why Time Series Is Different

Time series มี order:

~~~text
past → present → future
~~~

sample ไม่ exchangeable เหมือน tabular IID setting

ถ้า shuffle ก่อน split:

~~~text
future → training
past   → validation
~~~

model อาจเห็นข้อมูลอนาคตทางอ้อม

## 2. Learning Objectives

เมื่อจบบทนี้คุณควร:

- define timestamp/index/frequency
- inspect trend / seasonality / noise
- build naive and seasonal-naive baselines
- calculate lag features
- calculate rolling features without leakage
- explain autocorrelation
- explain stationarity
- understand differencing
- explain AR / MA / ARMA / ARIMA / SARIMA
- implement autoregression from scratch
- use statsmodels ARIMA
- understand walk-forward validation
- explain Prophet components
- explain ML/DL forecasting framing
- choose forecasting metrics carefully

## 3. Components

### Trend
long-term direction

### Seasonality
repeating pattern with known-ish period

### Cycles
longer irregular oscillations

### Noise
unexplained variation

conceptual:

~~~text
y_t = trend_t + seasonal_t + residual_t
~~~

หรือ multiplicative formulation ในบาง data

## 4. Frequency

DatetimeIndex ควรมี frequency semantics

ตัวอย่าง:
- hourly
- daily
- weekly
- monthly

missing timestamps ต้องแยกจาก missing values

## 5. Naive Baseline

one-step:

~~~text
ŷ_t = y_(t-1)
~~~

ถ้า model ชนะ naive ไม่ได้ ต้อง debug ก่อน

## 6. Seasonal Naive

ถ้า season length = s:

~~~text
ŷ_t = y_(t-s)
~~~

ตัวอย่าง daily data with weekly seasonality:

~~~text
s = 7
~~~

## 7. Forecast Horizon

อย่าพูดว่า model “ดี” โดยไม่ระบุ horizon

~~~text
h = 1
h = 24
h = 30
~~~

ความยากต่างกัน

## 8. Autocorrelation

ACF lag k:

~~~text
corr(y_t, y_(t-k))
~~~

ช่วยดู dependence across lags

PACF พยายามวัด direct association ของ lag k หลัง account intermediate lags

ใช้เป็น diagnostic ไม่ใช่ automatic truth

## 9. Stationarity

weak stationarity โดยย่อ:
- mean ไม่เปลี่ยนตามเวลา
- variance คงที่
- covariance ขึ้นกับ lag มากกว่า absolute time

ARIMA classical theory พึ่ง stationarity หลัง differencing สำหรับ ARMA component

## 10. Differencing

first difference:

~~~text
Δy_t = y_t - y_(t-1)
~~~

second difference:

~~~text
Δ²y_t = Δy_t - Δy_(t-1)
~~~

อย่า difference มากเกิน เพราะ signal อาจถูกทำลาย

## 11. AR(p)

Autoregressive model:

~~~text
y_t
=
c
+
φ₁ y_(t-1)
+
...
+
φ_p y_(t-p)
+
ε_t
~~~

มันคือ regression บน lagged target

## 12. MA(q)

Moving Average ใน ARIMA หมายถึง error terms:

~~~text
y_t
=
μ
+
ε_t
+
θ₁ ε_(t-1)
+
...
+
θ_q ε_(t-q)
~~~

ไม่ใช่ rolling arithmetic average แบบทั่วไป

## 13. ARMA

combine AR + MA สำหรับ stationary series

~~~text
ARMA(p,q)
~~~

## 14. ARIMA

~~~text
ARIMA(p,d,q)
~~~

- p = AR order
- d = differencing order
- q = MA order

statsmodels current ARIMA interface ยังใช้ order=(p,d,q)

## 15. SARIMA

seasonal extension:

~~~text
(p,d,q) × (P,D,Q,s)
~~~

statsmodels ARIMA interface รองรับ seasonal_order ด้วย

## 16. Lag Matrix

สำหรับ p=3:

~~~text
target y_t
features:
y_(t-1), y_(t-2), y_(t-3)
~~~

ห้ามให้ y_t หรือ future lags หลุดเข้า features

## 17. From-Scratch AutoRegression

src/autoregression.py มี:
- make_lag_matrix
- AutoRegressorOLS
- recursive forecast
- MAE/RMSE helpers

ใช้ least squares เพื่อเชื่อม Regression กับ Time Series

## 18. One-Step vs Recursive Forecast

### One-step
ใช้ observed lag ล่าสุดทุกครั้ง

### Recursive multi-step
forecast ก่อนหน้า feed กลับเป็น input

error จึงสะสม

## 19. Direct Multi-Horizon

สร้าง model แยกสำหรับแต่ละ horizon:

~~~text
h=1 model
h=2 model
...
~~~

หรือ multi-output model

trade-off กับ recursive strategy

## 20. Rolling Features

เช่น rolling mean:

~~~text
mean(y_(t-1),...,y_(t-7))
~~~

ต้อง shift ก่อน rolling เพื่อไม่รวม current/future target

ผิด:

~~~text
rolling mean includes y_t
~~~

ถูก:

~~~text
shift(1) → rolling(...)
~~~

## 21. Walk-Forward Validation

แทน random K-fold:

~~~text
Train [1..100] → Validate [101..120]
Train [1..120] → Validate [121..140]
Train [1..140] → Validate [141..160]
~~~

expanding window

หรือ rolling window ถ้า old history ไม่ relevant

## 22. Metrics

### MAE
units เดิม ตีความง่าย

### RMSE
penalize large errors

### MAPE
มีปัญหาเมื่อ actual ใกล้ 0

### sMAPE
ลดบางปัญหาแต่ยังมี edge cases

### MASE
scale-free เทียบ naive benchmark

metric ต้องสัมพันธ์ business/domain cost

## 23. Prediction Intervals

point forecast อย่างเดียวไม่บอก uncertainty

ต้องแยก:
- confidence in parameter/model
- forecast uncertainty
- distribution assumptions

ARIMA/Prophet บาง APIs ให้ intervals แต่ interpretation ขึ้นกับ model assumptions

## 24. Prophet

Prophet ใช้ additive model components เช่น:
- trend
- seasonality
- holidays/events

Python API ใช้ dataframe columns:
- ds = timestamp
- y = target

Prophet เหมาะเมื่อมี business-like seasonality/trend/change points แต่ไม่ใช่ winner universal

ติดตั้ง optional:

~~~bash
python -m pip install -r requirements-batch06-extras.txt
~~~

## 25. ML Forecasting

แปลง time series เป็น supervised table:

~~~text
lags
rolling stats
calendar features
external regressors
→ model
~~~

models:
- Linear/Ridge
- Random Forest
- Gradient Boosting
- XGBoost/LightGBM/CatBoost

ต้อง generate features แบบ causal

## 26. DL Forecasting

แนวคิด:
- MLP on lag windows
- RNN/LSTM/GRU
- Temporal CNN
- Transformers

แต่ deep model ไม่ได้ชนะเสมอโดยเฉพาะ data น้อย

Chapter 18 จะเริ่ม neural foundations ก่อน

## 27. Exogenous Variables

external regressors:
- weather
- promotions
- price
- holidays

ถามสำคัญ:

> ตอน forecast future เรารู้ feature นี้จริงไหม?

ถ้าไม่รู้ ต้อง forecast exogenous variable หรือใช้ scenario assumptions

## 28. Leakage Examples

1. random split
2. future rolling window
3. global scaler fit full timeline
4. future target-derived feature
5. revision data ที่ตอนอดีตยังไม่รู้
6. exogenous future value ที่ production ไม่มี

## 29. Common Mistakes

1. no naive baseline
2. random train_test_split
3. MAPE near zero
4. difference blindly
5. assume ACF implies causality
6. tune on final future test
7. use future exogenous information
8. report one horizon only without saying h
9. fit scaler on full timeline
10. ignore regime changes

## 30. Exercises / Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 31. Checklist

- [ ] trend/seasonality
- [ ] naive baselines
- [ ] lags
- [ ] ACF/PACF
- [ ] stationarity
- [ ] differencing
- [ ] ARIMA/SARIMA
- [ ] walk-forward
- [ ] metrics
- [ ] Prophet concepts
- [ ] ML/DL framing
- [ ] no future leakage

## 32. What's Next

Chapter 18 จะสร้าง neuron, perceptron, linear layer, activations และ MLP forward pass จากศูนย์
