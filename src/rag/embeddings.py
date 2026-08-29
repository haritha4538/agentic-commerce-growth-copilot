"""
Thin wrapper around the Gemini embeddings API (via the `google-genai`
package). This is the ONLY module that talks to the embedding API — ingest.py
and retriever.py call into it rather than using the SDK directly, so there is
one place to change if the SDK surface changes.

No API key is ever hardcoded here — it's read from src.config, which in turn
reads it from the environment / .env file (see src/config.py, Phase 1).
"""

from __future__ import annotations

from typing import List, Optional

from src.config import require_gemini_key, GEMINI_EMBEDDING_MODEL

# Task types the Gemini embedding API distinguishes between. Documents get
# indexed with RETRIEVAL_DOCUMENT; user queries get embedded with
# RETRIEVAL_QUERY. Using matched task types measurably improves retrieval
# quality over using one generic embedding type for both.
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


class EmbeddingGenerationError(Exception):
    """Raised when the Gemini embedding API cannot be reached or returns an
    unusable response. Callers should catch this and surface a clear
    message rather than letting a raw SDK traceback bubble up."""


_client = None  # lazy singleton; created on first real use


def _get_client():
    """
    Creates (once) and returns a google-genai Client using the API key from
    src.config. Raises EnvironmentError via require_gemini_key() if the key
    is missing — deliberately loud and early rather than failing deep
    inside an SDK call with a confusing error.
    """
    global _client
    if _client is not None:
        return _client

    api_key = require_gemini_key()  # raises EnvironmentError if unset

    try:
        from google import genai
    except ImportError as exc:
        raise EmbeddingGenerationError(
            "The 'google-genai' package is not installed. Run "
            "`pip install -r requirements.txt` and try again."
        ) from exc

    try:
        _client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingGenerationError(f"Failed to initialize the Gemini client: {exc}") from exc

    return _client


def _extract_vectors(response) -> List[List[float]]:
    """
    Normalizes the embed_content() response into a plain list of float
    vectors. Written defensively because the google-genai SDK's response
    shape has changed across versions (`.embeddings[i].values` vs a flatter
    structure) — this isolates that fragility to one place.
    """
    embeddings_attr = getattr(response, "embeddings", None)
    if embeddings_attr is None:
        raise EmbeddingGenerationError(
            "Unexpected response shape from Gemini embed_content() — no 'embeddings' field found."
        )

    vectors = []
    for item in embeddings_attr:
        values = getattr(item, "values", None)
        if values is None and isinstance(item, dict):
            values = item.get("values")
        if values is None:
            raise EmbeddingGenerationError(
                "Unexpected embedding item shape from Gemini — no 'values' field found."
            )
        vectors.append(list(values))
    return vectors


def embed_texts(
    texts: List[str],
    task_type: str = TASK_TYPE_DOCUMENT,
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts.

    Args:
        texts: non-empty strings to embed. Empty/whitespace-only entries
               are rejected up front with a clear error rather than being
               silently sent to the API.
        task_type: TASK_TYPE_DOCUMENT (default) or TASK_TYPE_QUERY.
        model: overrides GEMINI_EMBEDDING_MODEL from config if provided.

    Returns:
        A list of embedding vectors, one per input text, in the same order.

    Raises:
        EnvironmentError: if GEMINI_API_KEY is not configured.
        EmbeddingGenerationError: for any SDK/API failure, empty input, or
            malformed response.
    """
    if not texts:
        raise EmbeddingGenerationError("embed_texts() was called with an empty list of texts.")

    cleaned = [t.strip() for t in texts]
    if any(not t for t in cleaned):
        raise EmbeddingGenerationError(
            "embed_texts() received one or more empty/whitespace-only text entries."
        )

    client = _get_client()
    model_name = model or GEMINI_EMBEDDING_MODEL

    try:
        from google.genai import types
        config = types.EmbedContentConfig(task_type=task_type)
        response = client.models.embed_content(model=model_name, contents=cleaned, config=config)
    except ImportError:
        # Older/newer SDK versions may not expose types.EmbedContentConfig;
        # fall back to calling without the optional config rather than failing.
        try:
            response = client.models.embed_content(model=model_name, contents=cleaned)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingGenerationError(f"Gemini embedding request failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingGenerationError(f"Gemini embedding request failed: {exc}") from exc

    vectors = _extract_vectors(response)
    if len(vectors) != len(cleaned):
        raise EmbeddingGenerationError(
            f"Gemini returned {len(vectors)} embeddings for {len(cleaned)} input texts."
        )
    return vectors


def embed_query(text: str, model: Optional[str] = None) -> List[float]:
    """Convenience wrapper for embedding a single user query."""
    if not text or not text.strip():
        raise EmbeddingGenerationError("embed_query() was called with an empty query string.")
    return embed_texts([text], task_type=TASK_TYPE_QUERY, model=model)[0]
