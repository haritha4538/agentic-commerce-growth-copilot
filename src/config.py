"""
Central configuration for the project.

Loads environment variables via python-dotenv and exposes path constants so
no other module hardcodes file paths or reads os.environ directly.

IMPORTANT: never hardcode API keys. GEMINI_API_KEY (used starting Phase 5)
must be set in a local .env file, which is gitignored. See .env.example.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
UPLOAD_DIR = DATA_DIR / "uploads"

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
POLICIES_DIR = KNOWLEDGE_BASE_DIR / "policies"
CHROMA_STORE_DIR = KNOWLEDGE_BASE_DIR / "chroma_store"

SAMPLE_CUSTOMERS_PATH = SAMPLE_DATA_DIR / "customers.csv"
SAMPLE_PRODUCTS_PATH = SAMPLE_DATA_DIR / "products.csv"
SAMPLE_ORDERS_PATH = SAMPLE_DATA_DIR / "orders.csv"

# Ensure runtime-writable directories exist (safe no-ops if already present)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment / secrets
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

APP_ENV = os.getenv("APP_ENV", "development")


def require_gemini_key() -> str:
    """
    Call this from any module that is about to make a Gemini API call.
    Fails loudly and early instead of silently sending a bad request.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key. "
            "Never hardcode API keys in source files."
        )
    return GEMINI_API_KEY
