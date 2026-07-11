"""Regression test for Chroma query embedding batch shape."""
from pathlib import Path

def test_retriever_embed_query_no_flatten_regression():
    source = (Path(__file__).parents[1] / "rag_pipeline" / "retriever.py").read_text()
    assert "return converted[0] if converted else []" not in source
    assert "return self._to_list_of_floats(raw)" in source
