"""
Node functions for the LangGraph orchestration graph.

Each function has the signature LangGraph expects: `(state: GraphState) ->
dict`, returning only the keys it updates. None of these functions import
or depend on LangGraph itself — they're plain, independently testable
Python functions that graph.py wires into a StateGraph. This also means
every node here can be called directly (as done in scripts/test_langgraph.py
and this module's own offline checks) without a LangGraph runtime.

Every node is defensive: a failure in one node is recorded as a warning/
error and an empty-but-well-shaped result, so downstream nodes degrade
gracefully instead of raising and killing the whole graph run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from src.orchestration.state import GraphState

from src.analytics.analytics_engine import run_full_analysis

from src.agents.strategy_agent import (
    build_business_context,
    retrieve_policy_context,
    _build_prompt,
    _call_gemini,
    _parse_json_response,
    _validate_and_normalize,
    StrategyGenerationError,
    REQUIRED_SECTIONS,
    LIST_SECTIONS,
)
from src.rag.embeddings import EmbeddingGenerationError
from src.rag.retriever import RetrievalError


def _log_entry(node: str, status: str, message: str) -> Dict[str, Any]:
    return {
        "node": node,
        "status": status,  # "success" | "warning" | "error" | "skipped"
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ==========================================================================
# 1. Analytics Agent
# ==========================================================================

def analytics_agent_node(state: GraphState) -> dict:
    """
    Consumes the three input DataFrames from state and runs the existing
    (Phase 2) Analytics Engine — no calculation logic lives here, this node
    only wires state in/out of run_full_analysis().
    """
    customers_df = state.get("customers_df")
    products_df = state.get("products_df")
    orders_df = state.get("orders_df")

    if customers_df is None or products_df is None or orders_df is None:
        msg = "Analytics Agent skipped: one or more input DataFrames are missing from state."
        return {
            "analytics_results": {},
            "growth_signals": {},
            "warnings": [msg],
            "errors": [msg],
            "node_execution_log": [_log_entry("analytics_agent", "error", msg)],
        }

    try:
        analytics_results = run_full_analysis(customers_df, products_df, orders_df)
    except Exception as exc:  # noqa: BLE001
        msg = f"Analytics Agent failed: {exc}"
        return {
            "analytics_results": {},
            "growth_signals": {},
            "warnings": [msg],
            "errors": [msg],
            "node_execution_log": [_log_entry("analytics_agent", "error", msg)],
        }

    growth_signals = analytics_results.get("growth_signals", {})
    kpis = analytics_results.get("executive_kpis", {})
    signal_counts = {k: len(v) for k, v in growth_signals.items()}
    msg = (
        f"Computed analytics — revenue={kpis.get('total_revenue')}, "
        f"orders={kpis.get('total_orders')}, growth_signals={signal_counts}"
    )

    return {
        "analytics_results": analytics_results,
        "growth_signals": growth_signals,
        "node_execution_log": [_log_entry("analytics_agent", "success", msg)],
    }


# ==========================================================================
# 2. Retrieval Agent
# ==========================================================================

def retrieval_agent_node(state: GraphState) -> dict:
    """
    Queries the existing RAG retriever (via the Phase 5
    retrieve_policy_context() helper, which runs one topic query per
    pricing/promotion/marketing/retention topic against
    src.rag.retriever.retrieve_relevant_chunks and deduplicates the
    results) and stores the retrieved chunks in state for the Strategy
    Agent node to consume — without re-querying.
    """
    try:
        chunks, sources_used, retrieval_warnings = retrieve_policy_context()
    except StrategyGenerationError as exc:
        # retrieve_policy_context() raises this when EVERY topic query
        # failed (e.g. the knowledge base was never ingested).
        msg = f"Retrieval Agent failed: {exc}"
        return {
            "retrieved_policy_chunks": [],
            "policy_sources_used": [],
            "warnings": [msg],
            "errors": [msg],
            "node_execution_log": [_log_entry("retrieval_agent", "error", msg)],
        }
    except (RetrievalError, EmbeddingGenerationError, EnvironmentError) as exc:
        msg = f"Retrieval Agent failed: {exc}"
        return {
            "retrieved_policy_chunks": [],
            "policy_sources_used": [],
            "warnings": [msg],
            "errors": [msg],
            "node_execution_log": [_log_entry("retrieval_agent", "error", msg)],
        }
    except Exception as exc:  # noqa: BLE001
        msg = f"Retrieval Agent failed unexpectedly: {exc}"
        return {
            "retrieved_policy_chunks": [],
            "policy_sources_used": [],
            "warnings": [msg],
            "errors": [msg],
            "node_execution_log": [_log_entry("retrieval_agent", "error", msg)],
        }

    msg = f"Retrieved {len(chunks)} policy chunk(s) from {len(sources_used)} source(s): {sources_used}"
    status = "success" if chunks else "warning"

    return {
        "retrieved_policy_chunks": chunks,
        "policy_sources_used": sources_used,
        "warnings": retrieval_warnings,
        "node_execution_log": [_log_entry("retrieval_agent", status, msg)],
    }


# ==========================================================================
# 3. Strategy Agent
# ==========================================================================

def strategy_agent_node(state: GraphState) -> dict:
    """
    Reuses the exact building blocks Phase 5's generate_strategy_report()
    uses internally (build_business_context, _build_prompt, _call_gemini,
    _parse_json_response, _validate_and_normalize) rather than calling
    generate_strategy_report() itself. That function does its own policy
    retrieval, which would mean re-embedding and re-querying the knowledge
    base a second time in this graph — wasteful now that the Retrieval
    Agent node has already done it. Composing the same lower-level
    functions here keeps a single source of truth for prompt/parsing logic
    without touching src/agents/strategy_agent.py.
    """
    analytics_results = state.get("analytics_results") or {}
    growth_signals = state.get("growth_signals") or {}
    policy_chunks = state.get("retrieved_policy_chunks") or []

    if not analytics_results:
        msg = "Strategy Agent skipped: no analytics_results in state (Analytics Agent may have failed)."
        return {
            "strategy_report": {},
            "raw_model_output": "",
            "warnings": [msg],
            "node_execution_log": [_log_entry("strategy_agent", "skipped", msg)],
        }

    try:
        business_context = build_business_context(analytics_results, growth_signals)
        prompt = _build_prompt(business_context, policy_chunks)
        raw_output = _call_gemini(prompt)
        parsed = _parse_json_response(raw_output)
        normalized_report, structural_warnings = _validate_and_normalize(parsed)
    except (StrategyGenerationError, EnvironmentError, EmbeddingGenerationError) as exc:
        msg = f"Strategy Agent failed: {exc}"
        return {
            "strategy_report": {},
            "raw_model_output": "",
            "warnings": [msg],
            "errors": [msg],
            "node_execution_log": [_log_entry("strategy_agent", "error", msg)],
        }
    except Exception as exc:  # noqa: BLE001
        msg = f"Strategy Agent failed unexpectedly: {exc}"
        return {
            "strategy_report": {},
            "raw_model_output": "",
            "warnings": [msg],
            "errors": [msg],
            "node_execution_log": [_log_entry("strategy_agent", "error", msg)],
        }

    msg = f"Generated strategy report with {len(normalized_report)} section(s)."
    return {
        "strategy_report": normalized_report,
        "raw_model_output": raw_output,
        "warnings": structural_warnings,
        "node_execution_log": [_log_entry("strategy_agent", "success", msg)],
    }


# ==========================================================================
# 4. Validator Agent
# ==========================================================================

def validator_agent_node(state: GraphState) -> dict:
    """
    Structural + citation validation of the strategy report:
      - confirms all REQUIRED_SECTIONS keys are present
      - confirms at least some recommendations cite a policy source, and
        flags any citation that doesn't match a source actually retrieved
        this run (a hallucinated citation)

    Per the Phase 6 spec, this node prefers warnings over hard failure:
    `validation_results["is_valid"]` is only False in the genuinely fatal
    case of a missing/empty report. Every lesser issue (a missing section,
    zero citations, an unrecognized citation) is recorded as a warning so
    the rest of the pipeline still produces a usable, inspectable result.
    """
    report = state.get("strategy_report") or {}
    known_sources = set(state.get("policy_sources_used") or [])
    warnings: List[str] = []

    section_presence = {section: (section in report) for section in REQUIRED_SECTIONS}
    all_sections_present = all(section_presence.values())
    if not all_sections_present:
        missing = [s for s, present in section_presence.items() if not present]
        warnings.append(f"Validator: missing required section(s) in strategy report: {missing}")

    total_citations = 0
    cited_sources = set()
    unrecognized_citations = set()
    sections_with_citations = []

    for section in LIST_SECTIONS:
        items = report.get(section) or []
        if not isinstance(items, list):
            continue
        section_has_citation = False
        for item in items:
            if not isinstance(item, dict):
                continue
            ref = item.get("policy_reference")
            if ref:
                total_citations += 1
                cited_sources.add(ref)
                section_has_citation = True
                if ref not in known_sources:
                    unrecognized_citations.add(ref)
        if section_has_citation:
            sections_with_citations.append(section)

    if total_citations == 0:
        warnings.append(
            "Validator: no policy citations found anywhere in the strategy report — "
            "recommendations may not be grounded in company policy."
        )
    if unrecognized_citations:
        warnings.append(
            "Validator: strategy report cites policy source(s) not among the chunks "
            f"retrieved this run (possible hallucinated citation): {sorted(unrecognized_citations)}"
        )

    has_summary = isinstance(report.get("executive_summary"), str) and bool(report.get("executive_summary", "").strip())
    is_valid = bool(report) and has_summary
    if not is_valid:
        warnings.append("Validator: strategy report is empty or missing an executive summary — treating as invalid.")

    validation_results = {
        "required_sections_present": section_presence,
        "all_sections_present": all_sections_present,
        "citation_summary": {
            "total_citations": total_citations,
            "cited_sources": sorted(cited_sources),
            "sections_with_citations": sections_with_citations,
        },
        "unrecognized_citations": sorted(unrecognized_citations),
        "is_valid": is_valid,
    }

    status = "success" if is_valid and not warnings else ("warning" if is_valid else "error")
    msg = (
        f"Validation complete — all_sections_present={all_sections_present}, "
        f"total_citations={total_citations}, is_valid={is_valid}"
    )

    return {
        "validation_results": validation_results,
        "warnings": warnings,
        "node_execution_log": [_log_entry("validator_agent", status, msg)],
    }
