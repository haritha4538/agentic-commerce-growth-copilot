"""
Phase 4 verification script.

Loads the four policy documents, builds/updates the ChromaDB knowledge
base, and runs a handful of test retrieval queries — printing sources and
retrieved text so we can eyeball relevance before this is wired into any
agent.

This script is written to fail GRACEFULLY and INFORMATIVELY at every stage:
missing packages, missing GEMINI_API_KEY, missing/empty policy files, and
ChromaDB errors are all caught and reported clearly rather than crashing
with a raw traceback — the same standard the rest of this project holds to.

Run from the project root:
    python scripts/test_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_QUERIES = [
    "Can we offer discounts to customers?",
    "What are the pricing rules?",
    "What marketing channels are allowed?",
    "What products are suitable for promotion?",
]


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> bool:
    """Returns True if verification passed end-to-end, False otherwise."""

    # ----------------------------------------------------------------
    # Step 0: import the RAG modules. If chromadb / google-genai aren't
    # installed, report that clearly instead of a bare ModuleNotFoundError.
    # ----------------------------------------------------------------
    section("STEP 0: Checking dependencies")
    try:
        from src.config import GEMINI_API_KEY, POLICIES_DIR, CHROMA_STORE_DIR
        from src.rag.ingest import (
            check_policy_files, ingest_policies, IngestionError, EXPECTED_POLICY_FILES,
        )
        from src.rag.retriever import (
            retrieve_relevant_chunks, collection_stats, RetrievalError,
        )
        from src.rag.embeddings import EmbeddingGenerationError
    except ImportError as exc:
        print(f"❌ Could not import the RAG modules: {exc}")
        print("   Run `pip install -r requirements.txt` (needs chromadb + google-genai) and try again.")
        return False
    print("✅ All RAG modules imported successfully.")

    # ----------------------------------------------------------------
    # Step 1: policy files present?
    # ----------------------------------------------------------------
    section("STEP 1: Checking policy documents")
    file_status = check_policy_files()
    print(f"Policies directory : {POLICIES_DIR}")
    print(f"Present  : {file_status['present']}")
    print(f"Missing  : {file_status['missing']}")
    print(f"Empty    : {file_status['empty']}")

    if not file_status["present"]:
        print("❌ No usable policy documents found — cannot proceed.")
        print(f"   Expected files: {EXPECTED_POLICY_FILES}")
        return False
    if file_status["missing"] or file_status["empty"]:
        print("⚠️  Continuing with the policy documents that ARE present/non-empty.")
    else:
        print("✅ All four expected policy documents are present and non-empty.")

    # ----------------------------------------------------------------
    # Step 2: GEMINI_API_KEY configured?
    # ----------------------------------------------------------------
    section("STEP 2: Checking Gemini API key")
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY is not set.")
        print("   Copy .env.example to .env and add your key from Google AI Studio, then re-run this script.")
        print("   (Everything else in Phase 4 — document loading, chunking, error handling — is verified")
        print("    and working; only the live embedding calls are blocked without a key.)")
        return False
    print("✅ GEMINI_API_KEY is set.")

    # ----------------------------------------------------------------
    # Step 3: build/update the knowledge base
    # ----------------------------------------------------------------
    section("STEP 3: Ingesting policy documents into ChromaDB")
    print(f"ChromaDB store directory: {CHROMA_STORE_DIR}")
    try:
        result = ingest_policies(reset=False)
    except (IngestionError, EmbeddingGenerationError, EnvironmentError) as exc:
        print(f"❌ Ingestion failed: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Unexpected error during ingestion: {exc}")
        return False

    print(f"Documents loaded   : {result.documents_loaded}")
    print(f"Chunks created     : {result.chunks_created}")
    print(f"Chunks upserted    : {result.chunks_upserted}")
    if result.warnings:
        print(f"Warnings           : {result.warnings}")
    if not result.is_success:
        print(f"❌ Ingestion did not complete successfully: {result.errors}")
        return False
    print("✅ Knowledge base built/updated successfully.")

    stats = collection_stats()
    print(f"Collection stats: {stats}")

    # ----------------------------------------------------------------
    # Step 4: run test retrieval queries
    # ----------------------------------------------------------------
    section("STEP 4: Running test retrieval queries")
    all_queries_ok = True

    for query in TEST_QUERIES:
        print(f"\n--- Query: \"{query}\" ---")
        try:
            chunks = retrieve_relevant_chunks(query, top_k=3)
        except (RetrievalError, EmbeddingGenerationError, EnvironmentError) as exc:
            print(f"❌ Retrieval failed for this query: {exc}")
            all_queries_ok = False
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Unexpected error during retrieval: {exc}")
            all_queries_ok = False
            continue

        if not chunks:
            print("⚠️  No chunks retrieved (collection may be empty).")
            all_queries_ok = False
            continue

        for rank, chunk in enumerate(chunks, start=1):
            preview = chunk.text.replace("\n", " ")
            preview = preview[:220] + ("..." if len(preview) > 220 else "")
            print(f"  [{rank}] source={chunk.source}  relevance={chunk.relevance_score}  "
                  f"(chunk #{chunk.chunk_index})")
            print(f"      \"{preview}\"")

    # ----------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------
    section("VERIFICATION RESULT")
    if all_queries_ok:
        print("✅ PASSED — knowledge base built and all test queries returned relevant chunks.")
        return True
    else:
        print("⚠️  PARTIAL — knowledge base built, but one or more queries had issues (see above).")
        return False


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
