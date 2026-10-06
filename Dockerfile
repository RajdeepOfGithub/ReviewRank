# ReviewRank scoring service. Build from the repository root:
#   docker build -t reviewrank:dev .
#   docker run --rm -p 8000:8000 reviewrank:dev
# The image contains only the serving code, the model file and its manifest (no data, no training code).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY artifacts/models/lgbm_primary.txt artifacts/models/lgbm_primary.txt
COPY artifacts/manifests/primary_model.json artifacts/manifests/primary_model.json

ENV REVIEWRANK_MODEL=/app/artifacts/models/lgbm_primary.txt \
    REVIEWRANK_MANIFEST=/app/artifacts/manifests/primary_model.json

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "reviewrank.serve:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
