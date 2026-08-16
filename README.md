# Insurance Claims Risk & Severity Platform

Production-oriented, end-to-end ML + MLOps + GenAI platform for predicting
insurance claim severity, explaining predictions in plain language, and
serving predictions via a batch CLI, a Streamlit dashboard, and a FastAPI
service — with monitoring, data validation, and responsible-AI guardrails
built in.

## What it demonstrates

- Public Allstate-style claims severity schema support (`loss` target).
- Data contracts and schema/quality validation (`src/validate.py`,
  `tests/test_data_validation.py`) before data reaches the pipeline.
- Robust preprocessing for numeric and categorical claim variables.
- Baselines, HistGradientBoosting and Random Forest comparison, with a
  safe log1p target transform (RMSLE-safe, see `tests/test_pipeline.py`).
- MLflow parameters, metrics, artifacts, and model registry-compatible
  logging.
- Batch scoring CLI (`src/score.py`) with model versioning, timestamping,
  and robust error handling.
- A FastAPI scoring service (`src/api.py`) with `/predict`, `/health`, and
  `/metadata` endpoints, fully unit-tested (`tests/test_api.py`).
- Lightweight production monitoring (`src/monitoring.py`): prediction
  statistics and drift heuristics vs. a reference baseline
  (`tests/test_monitoring.py`).
- A prompt-injection-resistant GenAI explanation layer (`src/genai.py`)
  that is deterministic by default and requires no API keys — see
  [Responsible AI](docs/responsible_ai.md).
- An optional Streamlit dashboard (`app.py`) and Docker packaging.
- CI on every push: linting, full pytest suite with coverage, and a
  synthetic-data smoke-train job (`.github/workflows/ci.yml`).

## Repository layout

```
src/            Core library: data, features, train, score, validate,
                api, monitoring, genai, models
tests/          Pytest suite (unit tests for every src/ module)
data/           Synthetic smoke-test sample only; real data is gitignored
configs/        Run configuration
docs/           Responsible AI notes and interview/design guide
notebooks/      Exploration notebooks (optional)
app.py          Streamlit dashboard
Dockerfile      Container image for the dashboard / batch jobs
```

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
python -m src.train --data data/train.csv --target loss --experiment claims-severity
python -m src.score --model artifacts/model.joblib --input data/new_claims.csv --output predictions.csv
streamlit run app.py
```

### Run the API service

```bash
uvicorn src.api:app --reload --port 8000
# then: GET /health, GET /metadata, POST /predict
```

### Run the tests

```bash
pytest -q --cov=src --cov-report=term-missing
```

### Docker

```bash
docker build -t claims-severity .
docker run -p 8501:8501 claims-severity
```

The image ships the Streamlit dashboard by default; override the container CMD to run `python -m src.train`, `python -m src.score`, or `uvicorn src.api:app` for batch/API jobs. Mount `data/` and `artifacts/` as volumes to persist inputs and trained models outside the container.

The repository does not bundle Kaggle data. Download the Allstate Claims Severity dataset from Kaggle, place `train.csv` in `data/`, and accept its license/terms. A synthetic smoke-test generator/sample is included (`src/data.py`, `data/smoke_sample.csv`) so CI and local smoke tests never require real or sensitive data.

## ML concepts

Severity is the conditional expected cost `E[loss | X]`. We use a log1p target because claim costs are positive and usually right-skewed. Metrics include MAE, RMSE, and RMSLE; a time-safe or claim-safe split should be used when timestamps or repeat claimants exist.

## MLOps

Run `mlflow ui --backend-store-uri mlruns` to inspect runs. The training script logs configuration, metrics, artifacts, and a serialized model. `src/monitoring.py` provides prediction and feature-drift heuristics that can be scheduled against live scoring output to catch model degradation early.

## Data validation

`src/validate.py` enforces a schema/data contract on incoming claims data (types, ranges, required fields) before scoring or training, and surfaces data-quality warnings that flow through to the GenAI explanation layer.

## GenAI safety

The explanation layer (`src/genai.py`) receives model outputs and feature contributions, but must never invent coverage decisions, diagnoses, legal conclusions, fraud determinations, or payout commitments. It is decision support only, not autonomous claims adjudication. Free-text claim fields are sanitized against prompt-injection attempts before being used in any prompt. See docs/responsible_ai.md for the full safety contract, model risk discussion, and operational safeguards.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push: dependency install, lint, the full pytest suite with coverage, and a smoke-train job against synthetic data — so the pipeline's core correctness is verified without any real or sensitive data ever touching CI.

See `docs/interview_guide.md` for definitions, foundations, trade-offs, and interview answers.
