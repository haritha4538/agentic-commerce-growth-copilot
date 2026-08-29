"""
Loads the policy markdown files, splits them into retrieval-sized chunks,
embeds them via Gemini (src.rag.embeddings), and persists them into a local
ChromaDB collection.

This module owns everything "upstream" of retrieval: file I/O, chunking,
and writing to the vector store. src/rag/retriever.py only reads from the
collection this module builds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config import POLICIES_DIR, CHROMA_STORE_DIR
from src.rag.embeddings import embed_texts, EmbeddingGenerationError, TASK_TYPE_DOCUMENT

COLLECTION_NAME = "policies"

# The four policy documents this project expects. Listed explicitly (rather
# than "load whatever's in the folder") so a missing file is detected and
# reported by name, not silently skipped.
EXPECTED_POLICY_FILES = [
    "pricing_policy.md",
    "marketing_policy.md",
    "promotion_guidelines.md",
    "product_information.md",
]

DEFAULT_CHUNK_SIZE = 900       # characters
DEFAULT_CHUNK_OVERLAP = 150    # characters


class IngestionError(Exception):
    """Raised for unrecoverable ingestion failures (e.g. no documents at all,
    or a ChromaDB error). Recoverable per-file issues are collected as
    warnings on IngestResult instead of raising."""


@dataclass
class IngestResult:
    is_success: bool = False
    documents_loaded: List[str] = field(default_factory=list)
    documents_missing: List[str] = field(default_factory=list)
    documents_empty: List[str] = field(default_factory=list)
    chunks_created: int = 0
    chunks_upserted: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "is_success": self.is_success,
            "documents_loaded": self.documents_loaded,
            "documents_missing": self.documents_missing,
            "documents_empty": self.documents_empty,
            "chunks_created": self.chunks_created,
            "chunks_upserted": self.chunks_upserted,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ==========================================================================
# 1. Document loading
# ==========================================================================

def load_policy_documents(policies_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Loads all expected policy markdown files from `policies_dir` (defaults
    to src.config.POLICIES_DIR).

    Returns a dict mapping filename -> raw text content. Missing files are
    silently omitted from the returned dict (not raised) — callers that
    need to know what's missing should use `check_policy_files()` instead,
    or inspect an IngestResult from `ingest_policies()`.
    """
    policies_dir = Path(policies_dir) if policies_dir else POLICIES_DIR
    documents: Dict[str, str] = {}

    if not policies_dir.exists():
        return documents

    for filename in EXPECTED_POLICY_FILES:
        file_path = policies_dir / filename
        if not file_path.exists():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        documents[filename] = text

    return documents


def check_policy_files(policies_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Returns {"present": [...], "missing": [...], "empty": [...]} for the
    expected policy files, without embedding or touching ChromaDB. Useful
    for a fast pre-flight check (e.g. in a future Streamlit "Knowledge
    Base" page).
    """
    policies_dir = Path(policies_dir) if policies_dir else POLICIES_DIR
    present, missing, empty = [], [], []

    for filename in EXPECTED_POLICY_FILES:
        file_path = policies_dir / filename
        if not file_path.exists():
            missing.append(filename)
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            missing.append(filename)
            continue
        if not text.strip():
            empty.append(filename)
        else:
            present.append(filename)

    return {"present": present, "missing": missing, "empty": empty}


# ==========================================================================
# 2. Chunking
# ==========================================================================

def _split_into_sections(text: str) -> List[str]:
    """
    Splits markdown text on '##' (H2) headers, keeping each header attached
    to its own section. Falls back to the whole text as one section if no
    H2 headers are present. This keeps each chunk topically coherent, which
    matters more for a handful of structured policy docs than a pure
    fixed-size sliding window would.
    """
    if not text.strip():
        return []

    # Split, keeping the "## " delimiter attached to the following section.
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    sections = [p.strip() for p in parts if p.strip()]
    return sections if sections else [text.strip()]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Splits `text` into overlapping chunks of at most `chunk_size`
    characters, preferring to break along markdown section (##) and
    paragraph boundaries before falling back to a hard character split.

    Returns an empty list for empty/whitespace-only input (never raises —
    the caller decides whether an empty document is an error).
    """
    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)  # guard against misconfiguration

    chunks: List[str] = []

    for section in _split_into_sections(text):
        if len(section) <= chunk_size:
            chunks.append(section)
            continue

        # Section too big for one chunk: split on paragraphs, then pack
        # paragraphs into chunks up to chunk_size with a character overlap
        # carried from the tail of the previous chunk.
        paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
                current = f"{overlap_tail}\n\n{paragraph}".strip()
            else:
                # A single paragraph longer than chunk_size: hard-split it.
                for i in range(0, len(paragraph), chunk_size - chunk_overlap):
                    chunks.append(paragraph[i:i + chunk_size])
                current = ""

        if current:
            chunks.append(current)

    return chunks


def build_chunk_records(documents: Dict[str, str], **chunk_kwargs) -> List[dict]:
    """
    Turns {filename: raw_text} into a flat list of chunk records ready for
    embedding + storage:
        {"id": "<filename>::chunk_<n>", "source": filename,
         "chunk_index": n, "text": chunk_text}
    """
    records = []
    for source, text in documents.items():
        chunks = chunk_text(text, **chunk_kwargs)
        for idx, chunk in enumerate(chunks):
            records.append({
                "id": f"{source}::chunk_{idx}",
                "source": source,
                "chunk_index": idx,
                "text": chunk,
            })
    return records


# ==========================================================================
# 3. ChromaDB persistence
# ==========================================================================

def _get_chroma_client(persist_directory: Optional[Path] = None):
    persist_directory = Path(persist_directory) if persist_directory else CHROMA_STORE_DIR
    persist_directory.mkdir(parents=True, exist_ok=True)
    try:
        import chromadb
    except ImportError as exc:
        raise IngestionError(
            "The 'chromadb' package is not installed. Run `pip install -r requirements.txt` and try again."
        ) from exc

    try:
        return chromadb.PersistentClient(path=str(persist_directory))
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Failed to open the ChromaDB store at {persist_directory}: {exc}") from exc


def ingest_policies(
    policies_dir: Optional[Path] = None,
    chroma_store_dir: Optional[Path] = None,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = False,
    embed_batch_size: int = 16,
) -> IngestResult:
    """
    End-to-end ingestion: load policy docs -> chunk -> embed via Gemini ->
    upsert into a persistent ChromaDB collection.

    Idempotent by design: chunk IDs are deterministic
    ("<filename>::chunk_<n>"), so re-running this after editing a policy
    file updates existing chunks in place via `upsert` rather than creating
    duplicates. Pass reset=True to wipe the collection and rebuild from
    scratch (e.g. after changing the chunking strategy).

    This function never raises for "expected" data problems (missing file,
    empty file) — those are reported in the returned IngestResult. It DOES
    raise IngestionError for infrastructure problems (ChromaDB unavailable)
    and lets EnvironmentError / EmbeddingGenerationError propagate from the
    embeddings module for missing-API-key / embedding-failure cases, since
    those block ingestion entirely and the caller needs to see them clearly.
    """
    result = IngestResult()

    file_status = check_policy_files(policies_dir)
    result.documents_missing = file_status["missing"]
    result.documents_empty = file_status["empty"]

    if file_status["missing"]:
        result.warnings.append(f"Missing policy file(s), skipped: {file_status['missing']}")
    if file_status["empty"]:
        result.warnings.append(f"Empty policy file(s), skipped: {file_status['empty']}")

    documents = load_policy_documents(policies_dir)
    # Drop empty documents (already flagged above) so we don't try to embed nothing.
    documents = {name: text for name, text in documents.items() if text.strip()}
    result.documents_loaded = list(documents.keys())

    if not documents:
        result.errors.append(
            "No usable policy documents were found — nothing to ingest. "
            f"Expected files in {policies_dir or POLICIES_DIR}: {EXPECTED_POLICY_FILES}"
        )
        return result

    records = build_chunk_records(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result.chunks_created = len(records)

    if not records:
        result.errors.append("Documents were loaded but produced zero chunks — check chunking parameters.")
        return result

    client = _get_chroma_client(chroma_store_dir)

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:  # noqa: BLE001
            pass  # collection may not exist yet on first run — that's fine

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Failed to get/create ChromaDB collection '{collection_name}': {exc}") from exc

    try:
        for start in range(0, len(records), embed_batch_size):
            batch = records[start:start + embed_batch_size]
            texts = [r["text"] for r in batch]
            vectors = embed_texts(texts, task_type=TASK_TYPE_DOCUMENT)

            collection.upsert(
                ids=[r["id"] for r in batch],
                embeddings=vectors,
                documents=texts,
                metadatas=[{"source": r["source"], "chunk_index": r["chunk_index"]} for r in batch],
            )
            result.chunks_upserted += len(batch)
    except EmbeddingGenerationError:
        raise  # let the caller see the real embedding error clearly
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Failed while writing chunks to ChromaDB: {exc}") from exc

    result.is_success = True
    return result
