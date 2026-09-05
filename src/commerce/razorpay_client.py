"""
Thin wrapper around the Razorpay Test Mode API — Payment Links only.

Mirrors the lazy-import + client-singleton pattern already used in
src/rag/embeddings.py. Credentials are read directly from the environment
(and Streamlit secrets, as a fallback) inside this module — src/config.py
is intentionally left untouched to minimize risk to existing code.

SAFETY GUARDRAIL: this module refuses to operate unless RAZORPAY_KEY_ID
looks like a Razorpay TEST key ("rzp_test_..."). A live key is rejected
outright, before any API call is attempted.

Scope, by design (per project instructions):
  - Payment Links only — no Orders API, no webhooks, no callback server.
  - No automatic/background status polling — fetch_payment_link_status()
    is a single, explicit, on-demand call triggered by a user action.
  - Fixed demo amount (₹499) — never derived from LLM-generated text.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional, Tuple

TEST_KEY_PREFIX = "rzp_test_"
FIXED_TEST_AMOUNT_INR = 499  # fixed per project scope — not derived from any AI-estimated figure


class RazorpayCredentialsError(Exception):
    """Raised when Razorpay credentials are missing, or don't look like Test Mode keys."""


class RazorpayClientError(Exception):
    """Raised when the Razorpay SDK is unavailable or an API call fails."""


def _get_credentials() -> Tuple[str, str]:
    """
    Reads RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET from the environment first,
    then falls back to Streamlit secrets if available. Never hardcoded.
    """
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")

    if not key_id or not key_secret:
        try:
            import streamlit as st
            key_id = key_id or str(st.secrets.get("RAZORPAY_KEY_ID", ""))
            key_secret = key_secret or str(st.secrets.get("RAZORPAY_KEY_SECRET", ""))
        except Exception:
            pass  # st.secrets unavailable outside a Streamlit run / no secrets.toml — that's fine

    if not key_id or not key_secret:
        raise RazorpayCredentialsError(
            "Razorpay credentials are not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET "
            "in your .env file (see .env.example) or in Streamlit secrets. Never hardcode credentials."
        )

    if not key_id.startswith(TEST_KEY_PREFIX):
        raise RazorpayCredentialsError(
            f"RAZORPAY_KEY_ID does not look like a Razorpay TEST key (expected it to start with "
            f"'{TEST_KEY_PREFIX}'). Refusing to proceed — this project only operates in Test Mode."
        )

    return key_id, key_secret


def is_test_mode_configured() -> bool:
    """Cheap, side-effect-free check for the UI: are valid-looking Test
    Mode credentials present? Never raises — returns False on any problem."""
    try:
        _get_credentials()
        return True
    except RazorpayCredentialsError:
        return False


_client = None  # lazy singleton, same pattern as src/rag/embeddings.py


def _get_client():
    global _client
    if _client is not None:
        return _client

    key_id, key_secret = _get_credentials()  # raises RazorpayCredentialsError

    try:
        import razorpay
    except ImportError as exc:
        raise RazorpayClientError(
            "The 'razorpay' package is not installed. Run `pip install -r requirements.txt` and try again."
        ) from exc

    try:
        _client = razorpay.Client(auth=(key_id, key_secret))
    except Exception as exc:  # noqa: BLE001
        raise RazorpayClientError(f"Failed to initialize the Razorpay client: {exc}") from exc

    return _client


def create_payment_link(description: str, reference_id: Optional[str] = None) -> dict:
    """
    Creates a Razorpay Test Mode Payment Link for the fixed demo amount
    (₹499 — see FIXED_TEST_AMOUNT_INR).

    Returns the raw Razorpay response dict (contains 'id', 'short_url',
    'status', etc.).

    Raises:
        RazorpayCredentialsError: credentials missing or not a test key.
        RazorpayClientError: SDK missing or the API call failed.
    """
    client = _get_client()
    reference_id = reference_id or f"growth-copilot-{uuid.uuid4().hex[:12]}"

    payload = {
        "amount": FIXED_TEST_AMOUNT_INR * 100,  # paise
        "currency": "INR",
        "description": description[:2048],
        "reference_id": reference_id,
        "reminder_enable": False,
    }

    try:
        response = client.payment_link.create(payload)
    except Exception as exc:  # noqa: BLE001 - covers razorpay.errors.* and network failures alike
        raise RazorpayClientError(f"Razorpay Payment Link creation failed: {exc}") from exc

    return response


def fetch_payment_link_status(payment_link_id: str) -> dict:
    """
    Fetches the current status of a previously created Payment Link.

    This is a single, explicit, user-triggered check — not a polling loop.
    No background thread, no automatic retry/refresh.
    """
    client = _get_client()
    try:
        return client.payment_link.fetch(payment_link_id)
    except Exception as exc:  # noqa: BLE001
        raise RazorpayClientError(f"Could not fetch Payment Link status: {exc}") from exc
