FROM node:16-bullseye AS ui-builder

WORKDIR /ui
COPY DigiCon_UI/package.json DigiCon_UI/package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY DigiCon_UI/ ./
RUN npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INPUT_PATH=/app/input \
    CHROMA_PATH=/app/chroma_db \
    GRAPH_PATH=/app/graph_db/qode_graph.json \
    KG_DATA_PATH=/app/data \
    INGEST_STATUS_FILE=/app/data/ingest_status.json

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_rag.txt /app/requirements_rag.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements_rag.txt

COPY . /app
COPY --from=ui-builder /ui/build /app/DigiCon_UI/build


RUN mkdir -p /app/input /app/data /app/chroma_db /app/graph_db /app/output && \
    chmod +x /app/docker-entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
