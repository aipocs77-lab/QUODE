#!/bin/sh
set -eu

echo "=================================================="
echo "QODE startup ingestion"
echo "=================================================="

python /app/startup_ingest.py

echo "=================================================="
echo "QODE ingestion complete. Starting API/UI on :8501"
echo "=================================================="

exec uvicorn api:app --host 0.0.0.0 --port 8501
