"""FastAPI entrypoint for DigiCon UI and the existing QODE chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_pipeline.chain import run_chain


APP_ROOT = Path(__file__).resolve().parent
UI_BUILD = APP_ROOT / "DigiCon_UI" / "build"
INPUT_DIR = Path(os.getenv("INPUT_PATH", "/app/input"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma_db")
GRAPH_PATH = os.getenv("GRAPH_PATH", "/app/graph_db/qode_graph.json")
STATUS_FILE = Path(os.getenv("INGEST_STATUS_FILE", "/app/data/ingest_status.json"))

app = FastAPI(title="QODE Neo4j API", version="1.0")
_history: list[dict] = []
_chain_lock = Lock()


class QueryRequest(BaseModel):
    question: str


def _ingest_status() -> dict:
    if not STATUS_FILE.exists():
        return {"status": "not_ready", "error": "ingestion status file is missing"}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "not_ready", "error": str(exc)}


def _questionnaire_path(status: dict) -> str | None:
    for item in status.get("processed", []):
        if item.get("kg_mode") == "qode_structured":
            return item.get("file")
    return None


@app.get("/health")
def health() -> dict:
    status = _ingest_status()
    return {
        "status": "ok" if status.get("status") == "ready" else "not_ready",
        "ingestion": status,
        "ui_build": UI_BUILD.exists(),
        "input_path": str(INPUT_DIR),
    }


@app.post("/query")
def query(request: QueryRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    status = _ingest_status()
    if status.get("status") != "ready":
        raise HTTPException(
            status_code=503,
            detail={
                "message": "QODE ingestion is not ready",
                "ingestion": status,
            },
        )

    try:
        with _chain_lock:
            result = run_chain(
                user_message=question,
                history=_history,
                excel_path=_questionnaire_path(status),
                chroma_path=CHROMA_PATH,
                graph_path=GRAPH_PATH,
                stream=False,
            )
            answer = result.get("text", "")
            _history.append({"role": "user", "content": question})
            _history.append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "diagram_path": result.get("diagram_path"),
            "diagram_type": result.get("diagram_type"),
            "mode": result.get("mode"),
            "eval_score": result.get("eval_score"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if (UI_BUILD / "static").exists():
    app.mount("/static", StaticFiles(directory=UI_BUILD / "static"), name="static")


@app.get("/{full_path:path}")
def react_app(full_path: str):
    requested = UI_BUILD / full_path
    if full_path and requested.is_file():
        return FileResponse(requested)

    index_file = UI_BUILD / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=503,
            detail="DigiCon UI build is unavailable",
        )
    return FileResponse(index_file)
