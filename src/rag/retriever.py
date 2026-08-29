"""
Retrieves relevant policy chunks from the ChromaDB collection built by
src/rag/ingest.py. This is the module the future Strategy Agent will call —
it is intentionally free of any LangGraph/agent-framework dependency so it
can be unit-tested and reused on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import CHROMA_STORE_DIR
from src.rag.embeddings import embed_query, EmbeddingGenerationError
from src.rag.ingest import COLLECTION_NAME


class RetrievalError(Exception):
    """Raised when retrieval cannot proceed — e.g. the collection hasn't
    been ingested yet, or ChromaDB itself failed. Callers should show the
    message to the user rather than a raw traceback."""


@dataclass
class RetrievedChunk:
    source: str            # policy filename, e.g. "pricing_policy.md"
    chunk_index: int
    text: str
    distance: Optional[float]        # raw ChromaDB distance (cosine: 0=identical, 2=opposite)
    relevance_score: Optional[float]  # derived similarity in [0, 1], higher = more relevant

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "distance": self.distance,
            "relevance_score": self.relevance_score,
        }


def _get_collection(chroma_store_dir: Optional[Path] = None, collection_name: str = COLLECTION_NAME):
    chroma_store_dir = Path(chroma_store_dir) if chroma_store_dir else CHROMA_STORE_DIR

    try:
        import chromadb
    except ImportError as exc:
        raise RetrievalError(
            "The 'chromadb' package is not installed. Run `pip install -r requirements.txt` and try again."
        ) from exc

    if not chroma_store_dir.exists():
        raise RetrievalError(
            f"No ChromaDB store found at {chroma_store_dir}. Run ingest_policies() "
            "(see scripts/test_rag.py) to build the knowledge base first."
        )

    try:
        client = chromadb.PersistentClient(path=str(chroma_store_dir))
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Failed to open the ChromaDB store at {chroma_store_dir}: {exc}") from exc

    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(
            f"Collection '{collection_name}' does not exist yet. Run ingest_policies() first. "
            f"(underlying error: {exc})"
        ) from exc


def _distance_to_relevance(distance: Optional[float]) -> Optional[float]:
    """
    Converts a cosine distance (0 = identical, 2 = opposite direction) into
    a more intuitive 0-1 relevance score (1 = most relevant). This is a
    linear rescaling, not a calibrated probability — treat it as a ranking
    aid, not a statistical confidence value.
    """
    if distance is None:
        return None
    similarity = 1 - (distance / 2)
    return round(max(0.0, min(1.0, similarity)), 4)


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 4,
    chroma_store_dir: Optional[Path] = None,
    collection_name: str = COLLECTION_NAME,
    source_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Embeds `query` via Gemini and returns the top_k most relevant policy
    chunks from the ChromaDB collection.

    Args:
        query: natural-language question, e.g. "Can we offer discounts to
               churn-risk customers?"
        top_k: number of chunks to return.
        source_filter: optional policy filename (e.g. "pricing_policy.md")
               to restrict retrieval to a single document.

    Returns:
        A list of RetrievedChunk, ordered most-relevant first. Returns an
        empty list (not an error) if the collection exists but has zero
        chunks matching the filter.

    Raises:
        EnvironmentError: if GEMINI_API_KEY is not configured.
        EmbeddingGenerationError: if the query embedding call fails.
        RetrievalError: if ChromaDB / the collection is unavailable.
    """
    if not query or not query.strip():
        raise RetrievalError("retrieve_relevant_chunks() requires a non-empty query string.")

    collection = _get_collection(chroma_store_dir, collection_name)

    count = collection.count()
    if count == 0:
        return []

    query_vector = embed_query(query)  # raises EnvironmentError / EmbeddingGenerationError upstream

    where_clause = {"source": source_filter} if source_filter else None

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            where=where_clause,
        )
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"ChromaDB query failed: {exc}") from exc

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[None] * len(documents)])[0]

    chunks = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(RetrievedChunk(
            source=metadata.get("source", "unknown") if metadata else "unknown",
            chunk_index=metadata.get("chunk_index", -1) if metadata else -1,
            text=text,
            distance=distance,
            relevance_score=_distance_to_relevance(distance),
        ))
    return chunks


def collection_stats(
    chroma_store_dir: Optional[Path] = None,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """
    Lightweight health-check helper: returns {"exists": bool, "chunk_count": int}
    without requiring a Gemini API call. Useful for a future Streamlit
    "Knowledge Base" page to show ingestion status without spending an
    embedding call just to check if the store is populated.
    """
    try:
        collection = _get_collection(chroma_store_dir, collection_name)
        return {"exists": True, "chunk_count": collection.count()}
    except RetrievalError:
        return {"exists": False, "chunk_count": 0}
