"""
Shared state object that flows through every node in the LangGraph
orchestration pipeline (src/orchestration/graph.py).

Uses TypedDict (not a dataclass) because LangGraph's StateGraph expects a
mapping-like schema it can apply partial updates to — each node function
returns only the keys it touches, and LangGraph merges them into the
running state.

Fields that multiple nodes append to (warnings, errors, node_execution_log)
are declared with `Annotated[..., operator.add]` so LangGraph concatenates
list updates across nodes instead of overwriting them. Every other field is
owned by exactly one node and is a plain overwrite.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import pandas as pd


class GraphState(TypedDict, total=False):
    # ---- inputs (set once, before the graph runs) ----
    customers_df: pd.DataFrame
    products_df: pd.DataFrame
    orders_df: pd.DataFrame

    # ---- populated by the Analytics Agent node ----
    analytics_results: Dict[str, Any]
    growth_signals: Dict[str, Any]

    # ---- populated by the Retrieval Agent node ----
    retrieved_policy_chunks: List[dict]
    policy_sources_used: List[str]

    # ---- populated by the Strategy Agent node ----
    strategy_report: Dict[str, Any]
    raw_model_output: str

    # ---- populated by the Validator Agent node ----
    validation_results: Dict[str, Any]

    # ---- accumulated across every node (see module docstring) ----
    warnings: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    node_execution_log: Annotated[List[dict], operator.add]


def create_initial_state(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> GraphState:
    """
    Builds a fully-populated starting state (every key present with an
    empty-but-well-typed default) so downstream nodes never need to guess
    whether a key exists yet — only whether it's still empty.
    """
    return GraphState(
        customers_df=customers_df,
        products_df=products_df,
        orders_df=orders_df,
        analytics_results={},
        growth_signals={},
        retrieved_policy_chunks=[],
        policy_sources_used=[],
        strategy_report={},
        raw_model_output="",
        validation_results={},
        warnings=[],
        errors=[],
        node_execution_log=[],
    )
