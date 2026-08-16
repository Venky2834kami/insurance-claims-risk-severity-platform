# Insurance Claims Risk & Severity Platform

Production-oriented claims severity ML + MLOps + GenAI explanation starter.

## What it demonstrates
- Public Allstate-style claims severity schema support (`loss` target).
- Robust preprocessing for numeric and categorical claim variables.
- Baselines, HistGradientBoosting and Random Forest comparison.
- MLflow parameters, metrics, artifacts, and model registry-compatible logging.
- Batch scoring CLI and optional Streamlit dashboard.
- GenAI-ready explanation prompt with a deterministic fallback.

## Quick start
```bash
pip install -r requirements.txt
python -m src.train --data data/train.csv --target loss --experiment claims-severity
python -m src.score --model artifacts/model.joblib --input data/new_claims.csv --output predictions.csv
streamlit run app.py
```

The repository does not bundle Kaggle data. Download the Allstate Claims Severity dataset from Kaggle, place `train.csv` in `data/`, and accept its license/terms. A synthetic smoke-test generator is included in `src/data.py`.

## ML concepts
Severity is the conditional expected cost `E[loss | X]`. We use a log1p target because claim costs are positive and usually right-skewed. Metrics include MAE, RMSE, and RMSLE; a time-safe or claim-safe split should be used when timestamps or repeat claimants exist.

## MLOps
Run `mlflow ui --backend-store-uri mlruns` to inspect runs. The training script logs configuration, metrics, artifacts, and a serialized model.

## GenAI safety
The explanation layer receives model outputs and feature contributions, but must not invent coverage decisions, diagnoses, legal conclusions, or payout commitments. It is decision support, not autonomous claims adjudication.

See `docs/interview_guide.md` for definitions, foundations, trade-offs, and interview answers.
