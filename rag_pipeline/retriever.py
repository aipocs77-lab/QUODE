"""
retriever.py — ChromaDB similarity retrieval for advanced-QODE.

"""

from __future__ import annotations

import logging

import chromadb

from .ingest import COLLECTION_NAME, _EMBED_MODEL, _get_embedding_function
from .embedding_cache import get_cached_embeddings, get_embedding_cache_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid diagram type values accepted by the metadata filter
# ---------------------------------------------------------------------------
DIAGRAM_TYPES = {"process", "people", "technology", "general"}

_COLLECTION_CACHE: dict[str, chromadb.Collection] = {}


def _get_collection(chroma_path: str) -> chromadb.Collection:
    if chroma_path in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[chroma_path]

    client = chromadb.PersistentClient(path=chroma_path)

    # Reuse the shared SentenceTransformer singleton from ingest.py (avoids 18s reload).
    # Wrap with an embedding cache but keep the same .name() so ChromaDB doesn't
    # raise a conflict against a collection created by the base SentenceTransformer EF.
    base_ef = _get_embedding_function()

    class CachedEmbeddingFunction:
        def name(self):
            return base_ef.name()

        def _to_list_of_floats(self, embeddings):
            result = []
            for emb in embeddings:
                if hasattr(emb, 'tolist'):
                    result.append(emb.tolist())
                elif isinstance(emb, list):
                    result.append([float(x) for x in emb])
                else:
                    result.append([float(emb)])
            return result

        def __call__(self, input):
            raw = get_cached_embeddings(
                texts=input,
                model=_EMBED_MODEL,
                embedding_fn=base_ef,
            )
            return self._to_list_of_floats(raw)

        def embed_query(self, input):
            """Return a BATCH of query embeddings as required by ChromaDB.

            Chroma passes query_texts as a sequence.  Returning converted[0]
            produced a flat list[float], which Chroma later interpreted as a
            batch whose individual items were floats, causing:
            "'float' object cannot be converted to 'Sequence'".
            """
            texts = input if isinstance(input, list) else [input]
            texts = [str(item) for item in texts]
            raw = get_cached_embeddings(
                texts=texts,
                model=_EMBED_MODEL,
                embedding_fn=base_ef,
            )
            return self._to_list_of_floats(raw)

    ef = CachedEmbeddingFunction()

    _COLLECTION_CACHE[chroma_path] = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _COLLECTION_CACHE[chroma_path]


def retrieve(
    query: str,
    k: int = 5,
    diagram_type: str | None = None,
    chroma_path: str = "./chroma_db",
) -> list[dict]:
    """Query ChromaDB and return the top-*k* most relevant documents.

    Args:
        query:        Natural-language search string.
        k:            Number of results to return.
        diagram_type: Optional filter — one of ``"process"``, ``"people"``,
                      ``"technology"``, or ``"general"``.  When provided, only
                      documents whose ``diagram_type`` metadata matches are
                      considered.
        chroma_path:  Path to the ChromaDB persistence directory.

    Returns:
        A list of dicts ``{"text": ..., "metadata": ..., "distance": ...}``,
        ordered from most to least similar.

    Note: Embeddings are cached in-memory (LRU, 1000 max). Cache hits reduce
    latency significantly on repeated queries.
    """
    collection = _get_collection(chroma_path)

    # Log cache stats periodically
    cache_info = get_embedding_cache_info()
    if cache_info["currsize"] > 0:
        logger.debug(
            "Embedding cache: %d entries, %.1f%% hit rate",
            cache_info["currsize"],
            cache_info["hit_rate"] * 100,
        )

    where: dict | None = None
    if diagram_type and diagram_type in DIAGRAM_TYPES:
        where = {"diagram_type": {"$eq": diagram_type}}

    query_kwargs: dict = {
        "query_texts": [query],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception as exc:
        # If the collection is empty or the filter yields no results, fall back
        # to an unfiltered query so the user still gets relevant context.
        logger.warning(
            "Filtered ChromaDB query failed (%s); retrying without filter.", exc
        )
        query_kwargs.pop("where", None)
        results = collection.query(**query_kwargs)

    docs: list[dict] = []
    if not results["documents"] or not results["documents"][0]:
        return docs

    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({"text": text, "metadata": meta, "distance": dist})

    return docs
