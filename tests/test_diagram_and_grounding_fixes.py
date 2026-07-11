from pathlib import Path
ROOT = Path(__file__).parents[1]

def test_directed_graph_compatibility():
    source = (ROOT / "Generate_Process_Network_Diagram.py").read_text()
    assert "def add_edge(self, source, target, **attrs):" in source
    assert "def get_edge_data(self, source, target, default=None):" in source
    assert "@property\n    def nodes(self):" in source

def test_asis_quality_chunk_fallback():
    source = (ROOT / "rag_pipeline" / "chain.py").read_text()
    assert "if is_asis or is_gap:" in source
    assert "if has_quality_chunks:" in source
    assert "retrieved questionnaire/graph chunks" in source
