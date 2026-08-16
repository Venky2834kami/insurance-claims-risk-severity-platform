# Interview Guide

## 30-second answer
I built an end-to-end claims severity platform. It loads tabular claims data, validates the target, imputes and encodes mixed feature types, log-transforms the positive and right-skewed loss target, compares a mean baseline with ensemble models, tracks runs in MLflow, serializes the winning pipeline, and exposes reproducible batch scoring plus a guarded GenAI explanation layer.

## Foundations
**Regression** predicts a continuous value. **Severity** is the monetary cost conditional on a claim occurring; frequency is how often claims occur. **EDA** means inspecting distributions, missingness, outliers, cardinality, leakage, and train/test drift before modeling. **Feature engineering** converts raw fields into stable predictive signals. **One-hot encoding** represents categories as binary columns. **Imputation** fills missing values using a rule learned only from training data. A **pipeline** keeps preprocessing and modeling together, preventing train/test contamination.

## Why log1p?
Claims losses are non-negative and commonly heavy-tailed. Modeling `log(1 + loss)` reduces the influence of extreme values and often improves relative-error behavior. Predictions are inverse-transformed with `expm1`, then clipped at zero. I report MAE for currency interpretability, RMSE to penalize large misses, and RMSLE for multiplicative error.

## Model choices
A mean predictor is a necessary baseline. Random Forest averages decorrelated trees and is robust but can be large. Histogram Gradient Boosting builds trees sequentially to correct residuals and is often a strong tabular baseline. I would tune with cross-validation, use early stopping where appropriate, and compare against a leakage-safe benchmark.

## MLOps answer
MLflow logs parameters, metrics, artifacts, and run metadata. The serialized artifact contains the fitted preprocessing and estimator, so batch scoring applies identical transformations. In production I would add a model registry, immutable data snapshots, CI tests, data validation, drift monitoring, champion/challenger deployment, rollback, access controls, and audit logs.

## GenAI answer
GenAI is not the predictor. The model produces the estimate; GenAI translates model output and approved feature-attribution evidence into a readable explanation. I would ground the prompt in structured inputs, constrain claims, redact PII, log prompts and responses, evaluate faithfulness and hallucination rates, and require human review for adverse or high-value decisions.

## Common questions
**How did you prevent leakage?** I fit imputers, encoders, and scalers inside a training pipeline and split before fitting. I would also remove post-settlement fields and use claim/time-group splits when appropriate.

**What if false negatives are costly?** Define the business loss first, use asymmetric thresholds or cost-sensitive learning, calibrate uncertainty, and evaluate operational metrics—not only RMSE.

**How would you improve it?** Add domain features, target-aware but leakage-safe validation, CatBoost/LightGBM if permitted, quantile or Tweedie objectives, SHAP explanations, monitoring, and a registry-backed deployment.

**How do you explain a prediction?** State the estimate, compare it with a baseline, show top validated contributors, disclose uncertainty and data limitations, and never claim causality from feature importance.
