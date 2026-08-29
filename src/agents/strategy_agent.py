"""
Strategy Agent — Phase 5

Takes the deterministic outputs of the Analytics Engine (Phase 2) and the
Growth Signals it computes, retrieves grounding policy context from the RAG
knowledge base (Phase 4), and asks Gemini to synthesize a structured,
business-focused strategy report.

HARD RULE this module is built around: the LLM never computes business
numbers. Every number the model is allowed to reference is handed to it
inside BUSINESS CONTEXT, built entirely from real Analytics Engine output.
The model's job is interpretation, prioritization, and policy-compliant
recommendation — not arithmetic. Full numeric-grounding validation of the
model's output against the source metrics is the job of the Validation
Layer (a later phase); this module does structural validation only (are
the required sections present and shaped correctly).

This module has no LangGraph dependency — it's a plain, testable Python
function that a future LangGraph node will simply call.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import require_gemini_key, GEMINI_MODEL_NAME
from src.rag.retriever import retrieve_relevant_chunks, RetrievalError
from src.rag.embeddings import EmbeddingGenerationError

REQUIRED_SECTIONS = [
    "executive_summary",
    "revenue_opportunities",
    "customer_growth_opportunities",
    "product_opportunities",
    "risks",
    "recommended_actions",
]

LIST_SECTIONS = [
    "revenue_opportunities",
    "customer_growth_opportunities",
    "product_opportunities",
    "risks",
    "recommended_actions",
]

# One retrieval query per business topic the Strategy Agent reasons about.
# Retrieving per-topic (rather than one generic query) gives broader, more
# relevant policy coverage across pricing/marketing/promotion/product docs.
TOPIC_QUERIES = {
    "pricing": "pricing rules, discount depth limits, and margin floors for growth opportunities",
    "promotion": "promotion guidelines for high-margin low-volume products, bundling, and seasonal campaigns",
    "marketing": "marketing channel investment and messaging guidelines",
    "retention": "customer retention and win-back policy for high-value or churn-risk customers",
}


class StrategyGenerationError(Exception):
    """Raised for unrecoverable failures generating a strategy report:
    Gemini API failures, unparseable output, or a completely unavailable
    knowledge base. Callers should show this message rather than a raw
    traceback."""


@dataclass
class StrategyReport:
    report: Dict[str, Any]                 # the six required sections
    business_context: Dict[str, Any]       # exact data handed to the model
    policy_context: List[dict]             # retrieved chunks used as grounding
    policy_sources_used: List[str]         # distinct policy filenames cited
    warnings: List[str] = field(default_factory=list)
    raw_model_output: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation of the full report + provenance."""
        return {
            "report": self.report,
            "business_context": self.business_context,
            "policy_sources_used": self.policy_sources_used,
            "warnings": self.warnings,
            # policy_context omitted by default (verbose); available via .policy_context
        }


# ==========================================================================
# Gemini client (mirrors src/rag/embeddings.py's pattern)
# ==========================================================================

_client = None  # lazy singleton


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = require_gemini_key()  # raises EnvironmentError if unset

    try:
        from google import genai
    except ImportError as exc:
        raise StrategyGenerationError(
            "The 'google-genai' package is not installed. Run "
            "`pip install -r requirements.txt` and try again."
        ) from exc

    try:
        _client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        raise StrategyGenerationError(f"Failed to initialize the Gemini client: {exc}") from exc

    return _client


def _call_gemini(prompt: str, model: Optional[str] = None) -> str:
    """
    Calls Gemini with a low temperature (deterministic, business-focused
    output rather than creative variation) and requests JSON output
    directly when the installed SDK version supports it. Falls back to a
    plain call (relying on prompt instructions for JSON-only output) if the
    structured-output config isn't available.
    """
    client = _get_client()
    model_name = model or GEMINI_MODEL_NAME

    try:
        from google.genai import types
        config = types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
        response = client.models.generate_content(model=model_name, contents=prompt, config=config)
    except ImportError:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
        except Exception as exc:  # noqa: BLE001
            raise StrategyGenerationError(f"Gemini generation request failed: {exc}") from exc
    except Exception:
        # The structured-output config path failed (unsupported field, SDK
        # mismatch, etc.) — retry once with a plain call before giving up.
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
        except Exception as exc2:  # noqa: BLE001
            raise StrategyGenerationError(f"Gemini generation request failed: {exc2}") from exc2

    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise StrategyGenerationError("Gemini returned an empty response.")
    return text


# ==========================================================================
# Business context construction (Analytics Engine output -> JSON-safe dict)
# ==========================================================================

def _sanitize_value(value: Any) -> Any:
    """Converts pandas/numpy scalar types into plain JSON-serializable
    Python types. Applied per-cell when turning DataFrames into records."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _df_records(
    df: Optional[pd.DataFrame],
    columns: Optional[List[str]] = None,
    n: Optional[int] = None,
) -> List[dict]:
    """Safely converts a DataFrame (or None/empty) into a list of
    JSON-safe record dicts. Never raises on missing/empty input."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    working = df[columns] if columns else df
    if n is not None:
        working = working.head(n)

    records = working.to_dict(orient="records")
    return [{k: _sanitize_value(v) for k, v in record.items()} for record in records]


def build_business_context(
    analytics_results: Dict[str, Any],
    growth_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts exactly the numbers the Strategy Agent is allowed to reason
    about from the full Analytics Engine bundle (src.analytics.analytics_engine.run_full_analysis).

    Deliberately a curated subset, not the full bundle: passing entire
    DataFrames (product_table, customer_table, rfm_table) to the LLM would
    blow up prompt size and dilute focus. Growth signals — the things most
    directly actionable — are passed through in full since they're already
    small, structured, and exactly what the agent should prioritize.
    """
    if growth_signals is None:
        growth_signals = analytics_results.get("growth_signals", {})

    kpis = analytics_results.get("executive_kpis", {}) or {}

    return {
        "executive_kpis": kpis,
        "top_categories_by_revenue": _df_records(analytics_results.get("revenue_by_category"), n=5),
        "channel_performance": _df_records(analytics_results.get("revenue_by_channel")),
        "top_products_by_revenue": _df_records(
            analytics_results.get("top_products"),
            columns=["product_id", "product_name", "category", "revenue", "units_sold", "margin_pct"],
            n=5,
        ),
        "bottom_products_by_revenue": _df_records(
            analytics_results.get("bottom_products"),
            columns=["product_id", "product_name", "category", "revenue", "units_sold", "margin_pct"],
            n=5,
        ),
        "customer_segment_counts": analytics_results.get("segment_counts", {}) or {},
        "growth_signals": growth_signals or {},
    }


# ==========================================================================
# Policy retrieval (RAG grounding)
# ==========================================================================

def retrieve_policy_context(
    top_k_per_topic: int = 3,
) -> Tuple[List[dict], List[str], List[str]]:
    """
    Runs one retrieval query per business topic (pricing, promotion,
    marketing, retention) against the Phase 4 knowledge base and returns a
    deduplicated set of chunks.

    Returns:
        (policy_chunks, distinct_sources_used, retrieval_warnings)

    Raises:
        EnvironmentError: if GEMINI_API_KEY is missing (propagates
            immediately — it would fail identically for every topic, so
            there's no value in catching it per-topic).
        StrategyGenerationError: if EVERY topic query fails (e.g. the
            knowledge base was never ingested) — there's nothing to ground
            the strategy report in, so generation cannot proceed.
    """
    all_chunks: List[dict] = []
    seen_keys = set()
    retrieval_warnings: List[str] = []
    successful_topics = 0

    for topic, query in TOPIC_QUERIES.items():
        try:
            chunks = retrieve_relevant_chunks(query, top_k=top_k_per_topic)
            successful_topics += 1
        except (RetrievalError, EmbeddingGenerationError) as exc:
            retrieval_warnings.append(f"Topic '{topic}': retrieval failed — {exc}")
            continue

        for chunk in chunks:
            key = (chunk.source, chunk.chunk_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_chunks.append({
                "topic": topic,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "relevance_score": chunk.relevance_score,
            })

    if successful_topics == 0:
        raise StrategyGenerationError(
            "Could not retrieve any policy context from the knowledge base. "
            "Make sure Phase 4 ingestion has been run (see scripts/test_rag.py) "
            f"and that GEMINI_API_KEY is valid. Details: {retrieval_warnings}"
        )

    sources_used = sorted({c["source"] for c in all_chunks})
    return all_chunks, sources_used, retrieval_warnings


def _format_policy_context(policy_chunks: List[dict]) -> str:
    if not policy_chunks:
        return "(No policy context was retrieved.)"
    blocks = []
    for chunk in policy_chunks:
        blocks.append(
            f"[source: {chunk['source']} | topic: {chunk['topic']} | relevance: {chunk['relevance_score']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


# ==========================================================================
# Prompt construction
# ==========================================================================

_RESPONSE_SCHEMA = """Return ONLY a single valid JSON object (no markdown code fences, no commentary before or after it) with EXACTLY these top-level keys:

{
  "executive_summary": "3-5 sentence plain-language summary of the current business state and the single biggest opportunity",
  "revenue_opportunities": [
    {"title": "...", "description": "...", "supporting_metric": "the specific number/fact from BUSINESS CONTEXT this is based on", "policy_reference": "a filename from POLICY CONTEXT, or null", "estimated_impact": "a hypothesis phrased with the word 'estimated', or null"}
  ],
  "customer_growth_opportunities": [ {"title": "...", "description": "...", "supporting_metric": "...", "policy_reference": "... or null", "estimated_impact": "... or null"} ],
  "product_opportunities": [ {"title": "...", "description": "...", "supporting_metric": "...", "policy_reference": "... or null", "estimated_impact": "... or null"} ],
  "risks": [
    {"risk": "short risk name", "description": "...", "supporting_metric": "..."}
  ],
  "recommended_actions": [
    {"action": "a concrete next step", "rationale": "...", "supporting_metric": "...", "policy_reference": "... or null", "priority": "High or Medium or Low", "estimated_impact": "... or null"}
  ]
}
"""


def _build_prompt(business_context: Dict[str, Any], policy_chunks: List[dict]) -> str:
    context_json = json.dumps(business_context, indent=2, default=str)
    policy_text = _format_policy_context(policy_chunks)

    return f"""You are the Strategy Agent inside an Agentic Commerce Growth Copilot — a business decision-support system, not a general-purpose chatbot.

STRICT RULES — follow every one of them:
1. Use ONLY the numbers given to you in BUSINESS CONTEXT. Never invent, estimate, or round a revenue/customer/product figure beyond what is provided or directly derivable from it by simple arithmetic.
2. Every recommendation must respect POLICY CONTEXT. When a recommendation follows a specific policy, set "policy_reference" to that policy's filename. Never recommend something a retrieved policy explicitly disallows (for example: discount depths above the authorized tier, stacking two percentage discounts, or discounting a thin-margin product).
3. Any forward-looking number (a projected revenue lift, a conversion improvement, etc.) is a hypothesis, not a fact. Its "estimated_impact" text must explicitly contain the word "estimated" and must never be phrased as a guaranteed outcome. If you cannot support a plausible estimate, use null instead of guessing.
4. Be specific and grounded in the actual data provided. Avoid generic advice that could apply to any company — every point should clearly trace back to a number or signal in BUSINESS CONTEXT.
5. Output ONLY the JSON object described below — no prose before or after it, no markdown fences.

BUSINESS CONTEXT (real, calculated data — the only numbers you may reference):
{context_json}

POLICY CONTEXT (retrieved from the company knowledge base — cite these filenames in policy_reference fields; do not cite a filename that isn't listed here):
{policy_text}

{_RESPONSE_SCHEMA}"""


# ==========================================================================
# Response parsing + structural validation
# ==========================================================================

def _parse_json_response(raw_text: str) -> Any:
    """Parses Gemini's raw text output into a Python object, tolerating
    markdown code fences that sometimes wrap JSON output despite
    instructions not to include them."""
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise StrategyGenerationError(
                "Gemini's response could not be parsed as JSON even after removing "
                f"markdown fences. Raw response (truncated): {raw_text[:500]!r}"
            ) from exc

    raise StrategyGenerationError(
        f"Gemini's response did not contain a JSON object. Raw response (truncated): {raw_text[:500]!r}"
    )


def _validate_and_normalize(parsed: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Structural validation only: confirms the six required sections exist
    and are shaped correctly (string for the summary, lists for the rest),
    coercing minor deviations (e.g. a single dict instead of a one-item
    list) rather than failing outright. This is NOT numeric-grounding
    validation — checking that supporting_metric values actually match
    real data is the job of the future Validation Layer phase.
    """
    warnings: List[str] = []

    if not isinstance(parsed, dict):
        raise StrategyGenerationError(
            f"Gemini's JSON response was not an object at the top level (got {type(parsed).__name__})."
        )

    normalized: Dict[str, Any] = {}

    summary = parsed.get("executive_summary")
    if not isinstance(summary, str) or not summary.strip():
        warnings.append("Missing or empty 'executive_summary' in model output.")
        summary = summary if isinstance(summary, str) else ""
    normalized["executive_summary"] = summary

    for section in LIST_SECTIONS:
        value = parsed.get(section)
        if value is None:
            warnings.append(f"Missing section '{section}' in model output — defaulted to an empty list.")
            normalized[section] = []
        elif isinstance(value, list):
            normalized[section] = value
        elif isinstance(value, dict):
            warnings.append(f"Section '{section}' was a single object, not a list — wrapped automatically.")
            normalized[section] = [value]
        else:
            warnings.append(
                f"Section '{section}' had an unexpected type ({type(value).__name__}) — defaulted to an empty list."
            )
            normalized[section] = []

    extra_keys = set(parsed.keys()) - set(REQUIRED_SECTIONS)
    if extra_keys:
        warnings.append(f"Model output included unexpected extra key(s), ignored: {sorted(extra_keys)}")

    return normalized, warnings


# ==========================================================================
# Public entry points
# ==========================================================================

def generate_strategy_report(
    analytics_results: Dict[str, Any],
    growth_signals: Optional[Dict[str, Any]] = None,
    top_k_policy_chunks_per_topic: int = 3,
    model: Optional[str] = None,
) -> StrategyReport:
    """
    Main Strategy Agent entry point.

    Args:
        analytics_results: the dict returned by
            src.analytics.analytics_engine.run_full_analysis().
        growth_signals: optional override; defaults to
            analytics_results["growth_signals"].
        top_k_policy_chunks_per_topic: chunks retrieved per policy topic
            (pricing/promotion/marketing/retention) before deduplication.
        model: optional override for GEMINI_MODEL_NAME.

    Returns:
        A StrategyReport with the six required sections, full provenance
        (business_context + policy_context used), and any structural
        warnings encountered while normalizing the model's output.

    Raises:
        EnvironmentError: GEMINI_API_KEY not configured.
        StrategyGenerationError: knowledge base unavailable, Gemini call
            failed, or the response could not be parsed/validated.
    """
    business_context = build_business_context(analytics_results, growth_signals)
    policy_chunks, sources_used, retrieval_warnings = retrieve_policy_context(
        top_k_per_topic=top_k_policy_chunks_per_topic
    )

    prompt = _build_prompt(business_context, policy_chunks)
    raw_output = _call_gemini(prompt, model=model)
    parsed = _parse_json_response(raw_output)
    normalized_report, structural_warnings = _validate_and_normalize(parsed)

    return StrategyReport(
        report=normalized_report,
        business_context=business_context,
        policy_context=policy_chunks,
        policy_sources_used=sources_used,
        warnings=retrieval_warnings + structural_warnings,
        raw_model_output=raw_output,
    )


def run_strategy_agent_from_dataframes(
    customers_df, products_df, orders_df,
    top_k_policy_chunks_per_topic: int = 3,
    model: Optional[str] = None,
) -> StrategyReport:
    """
    Convenience one-call entry point for scripts/Streamlit/a future
    LangGraph node: runs the Analytics Engine and the Strategy Agent
    together. Kept separate from generate_strategy_report() so the agent
    itself stays decoupled from raw DataFrames (matches the Phase 5 spec:
    "Accept analytics results and growth signals").
    """
    from src.analytics.analytics_engine import run_full_analysis

    analytics_results = run_full_analysis(customers_df, products_df, orders_df)
    return generate_strategy_report(
        analytics_results,
        growth_signals=analytics_results.get("growth_signals"),
        top_k_policy_chunks_per_topic=top_k_policy_chunks_per_topic,
        model=model,
    )
