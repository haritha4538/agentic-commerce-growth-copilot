"""
Builds and runs the LangGraph pipeline:

    START -> Analytics Agent -> Retrieval Agent -> Strategy Agent
          -> Validator Agent -> END

`langgraph` is imported lazily inside build_graph() (not at module top
level) so this module — and therefore src.orchestration.nodes, which has no
LangGraph dependency at all — stays importable even in environments where
`langgraph` isn't installed yet. The actual graph can only be built/run once
the package is available, which is exactly the same graceful-degradation
pattern used in src/rag and src/agents for chromadb/google-genai.
"""

from __future__ import annotations

from src.orchestration.state import GraphState, create_initial_state
from src.orchestration.nodes import (
    analytics_agent_node,
    retrieval_agent_node,
    strategy_agent_node,
    validator_agent_node,
)


class GraphBuildError(Exception):
    """Raised when the LangGraph pipeline cannot be built (e.g. the
    'langgraph' package is not installed)."""


_compiled_graph = None  # lazy singleton — build once, reuse across calls


def build_graph():
    """
    Constructs and compiles the four-node LangGraph pipeline. Cached after
    the first successful build so repeated calls (e.g. multiple
    invocations from a future Streamlit session) don't rebuild the graph
    every time.
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as exc:
        raise GraphBuildError(
            "The 'langgraph' package is not installed. Run `pip install -r requirements.txt` and try again."
        ) from exc

    workflow = StateGraph(GraphState)

    workflow.add_node("analytics_agent", analytics_agent_node)
    workflow.add_node("retrieval_agent", retrieval_agent_node)
    workflow.add_node("strategy_agent", strategy_agent_node)
    workflow.add_node("validator_agent", validator_agent_node)

    workflow.add_edge(START, "analytics_agent")
    workflow.add_edge("analytics_agent", "retrieval_agent")
    workflow.add_edge("retrieval_agent", "strategy_agent")
    workflow.add_edge("strategy_agent", "validator_agent")
    workflow.add_edge("validator_agent", END)

    _compiled_graph = workflow.compile()
    return _compiled_graph


def run_growth_copilot_graph(customers_df, products_df, orders_df) -> GraphState:
    """
    Runs the full four-agent pipeline once, start to finish, and returns
    the final merged state. This is the single entry point a future
    Streamlit "AI Strategy Center" page or CLI script should call — it
    hides graph construction entirely.

    Node-by-node execution is visible after the fact via
    final_state["node_execution_log"], which every node appends a
    `{node, status, message, timestamp}` entry to as it runs — no
    LangGraph streaming API needed for that visibility.
    """
    graph = build_graph()
    initial_state = create_initial_state(customers_df, products_df, orders_df)
    final_state = graph.invoke(initial_state)
    return final_state
