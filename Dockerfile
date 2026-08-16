FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scikit-learn/pandas wheels are prebuilt.
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app.py ./app.py
COPY configs ./configs

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# Non-root user for basic container hardening.
RUN useradd -m appuser
USER appuser

EXPOSE 8501

# Default: launch the Streamlit dashboard. Override CMD to run
# `python -m src.train` or `python -m src.score` for batch jobs.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
