# Mini Project — Forecasting Benchmark

Generate or use a dated series with:
- trend
- weekly seasonality
- noise
- optional anomalies

Compare:
1. naive
2. seasonal naive
3. from-scratch AR
4. statsmodels ARIMA
5. gradient boosting on lag/calendar features
6. optional Prophet

Protocol:
- chronological train/validation/test
- no full-series scaler
- tune only validation
- report horizon
- MAE/RMSE
- runtime
- forecast plot
- residual diagnostics

Explain which assumptions each model makes.
