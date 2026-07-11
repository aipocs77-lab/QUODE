import os
from pathlib import Path

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

INPUT_PATH = Path(os.getenv("INPUT_PATH", "/app/input"))
KG_DATA_PATH = Path(os.getenv("KG_DATA_PATH", "/app/data"))
METHODOLOGY_FILE = INPUT_PATH / "QODE_methodologies.md"
QODE_GRAPH_FILE = INPUT_PATH / "qode_graph.json"
WORKBOOK_FILE = INPUT_PATH / "sample_questions.xlsm"
