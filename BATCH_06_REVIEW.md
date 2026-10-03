# Batch 06 Review — Chapters 16–18

## Chapters

- 16 — Anomaly Detection
- 17 — Time Series Forecasting
- 18 — Neural Network Foundations

## Implemented From Scratch

### Anomaly
- ZScoreDetector
- RobustZScoreDetector
- MahalanobisDetector
- quantile thresholding

### Time Series
- lag matrix
- AutoRegressorOLS
- recursive forecast
- naive forecast
- seasonal naive
- MAE/RMSE

### Neural Foundations
- stable sigmoid
- ReLU
- tanh
- softmax
- MSE/BCE
- Linear layer
- Perceptron
- TinyMLP forward pass

## Framework Skills

- IsolationForest
- LocalOutlierFactor concepts
- OneClassSVM concepts
- statsmodels ARIMA
- sklearn LinearRegression on lags
- sklearn MLPRegressor
- optional Prophet

## Leakage Audit

- anomaly detector fit on training timeline only
- time split chronological
- lags use only past values
- MLP StandardScaler fit only on training lag rows
- validation selects model
- final test remains future holdout

## Neural Scope Audit

Chapter 18 intentionally does NOT teach full backpropagation.

It covers:
- values
- shapes
- layers
- activations
- loss
- computational graphs

Chapter 19 starts derivatives through the graph.

## Mastery Checklist

- [ ] compare anomaly definitions
- [ ] explain LOF novelty behavior
- [ ] explain anomaly threshold policy
- [ ] build chronological forecasting splits
- [ ] beat/check naive baseline
- [ ] explain ARIMA p,d,q
- [ ] build causal lag/rolling features
- [ ] compute neuron and dense layer forward
- [ ] trace MLP shapes
- [ ] explain why nonlinearity is required
- [ ] run Batch 06 integration project

## Exit Gate

ก่อน Chapter 19:

1. all Batch 06 tests pass
2. can identify leakage in anomaly/time-series pipelines
3. can derive AR lag regression
4. can trace every value shape in TinyMLP
5. can explain exactly what is still missing before an MLP can learn: gradients + backprop + optimizer
