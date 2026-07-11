"""Container startup: ingest runtime-mounted files before API startup."""
from __future__ import annotations
import json, logging, os, sys
from pathlib import Path

from rag_pipeline.generic_ingest import ingest_directory
from rag_pipeline.graph_retriever import invalidate_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("qode.startup")

INPUT_DIR = Path(os.getenv("INPUT_PATH", "/app/input"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma_db")
GRAPH_PATH = os.getenv("GRAPH_PATH", "/app/graph_db/qode_graph.json")
STATUS_FILE = Path(os.getenv("INGEST_STATUS_FILE", "/app/data/ingest_status.json"))

def write_status(data: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def main() -> int:
    try:
        result = ingest_directory(INPUT_DIR, CHROMA_PATH, GRAPH_PATH)
        invalidate_cache(graph_path=GRAPH_PATH, chroma_path=CHROMA_PATH)
        status = {"status": "ready", **result}
        write_status(status)
        logger.info("QODE READY: %s", json.dumps(status))
        return 0
    except Exception as exc:
        write_status({"status": "failed", "error": str(exc)})
        logger.exception("Startup ingestion failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
