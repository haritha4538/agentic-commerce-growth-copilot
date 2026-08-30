"""
Phase 6 verification script.

Loads sample data and executes the full LangGraph pipeline (Analytics ->
Retrieval -> Strategy -> Validator), printing each node's execution log
entry as it happened and the final validated report summary.

Like every prior verification script in this project, this one fails
gracefully and informatively at every stage rather than crashing with a
raw traceback.

Run from the project root:
    python scripts/test_langgraph.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_node_log(log: list):
    status_icons = {"success": "✅", "warning": "⚠️", "error": "❌", "skipped": "⏭️"}
    for entry in log:
        icon = status_icons.get(entry.get("status"), "•")
        print(f"{icon} [{entry.get('node')}] {entry.get('status').upper()} — {entry.get('message')}")


def main() -> bool:
    # ----------------------------------------------------------------
    section("STEP 0: Checking dependencies and imports")
    try:
        from src.config import GEMINI_API_KEY
        from src.data.loader import load_sample_data
        from src.rag.retriever import collection_stats
        from src.orchestration.graph import run_growth_copilot_graph, GraphBuildError
        from src.agents.strategy_agent import REQUIRED_SECTIONS
    except ImportError as exc:
        print(f"❌ Could not import required modules: {exc}")
        print("   Run `pip install -r requirements.txt` (needs langgraph, chromadb, google-genai) and try again.")
        return False
    print("✅ All modules imported successfully.")

    # ----------------------------------------------------------------
    section("STEP 1: Loading sample data")
    loaded = load_sample_data()
    if not loaded.is_valid:
        print("❌ Sample data failed validation:")
        print(loaded.summary())
        return False
    customers_df = loaded.dataframes["customers"]
    products_df = loaded.dataframes["products"]
    orders_df = loaded.dataframes["orders"]
    print(f"✅ Loaded {len(customers_df)} customers, {len(products_df)} products, {len(orders_df)} orders.")

    # ----------------------------------------------------------------
    section("STEP 2: Pre-flight checks (knowledge base + API key)")
    stats = collection_stats()
    print(f"Knowledge base status: {stats}")
    if not stats.get("exists") or stats.get("chunk_count", 0) == 0:
        print("❌ The policy knowledge base has not been ingested yet.")
        print("   Run `python scripts/test_rag.py` (Phase 4) first, then re-run this script.")
        return False
    print("✅ Knowledge base is populated.")

    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY is not set.")
        print("   Copy .env.example to .env and add your key, then re-run this script.")
        print("   (Graph construction and node wiring are implemented and covered by offline checks;")
        print("    only the live Gemini call inside the Strategy Agent node is blocked without a key.)")
        return False
    print("✅ GEMINI_API_KEY is set.")

    # ----------------------------------------------------------------
    section("STEP 3: Executing the LangGraph pipeline")
    try:
        final_state = run_growth_copilot_graph(customers_df, products_df, orders_df)
    except GraphBuildError as exc:
        print(f"❌ Could not build the graph: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Unexpected error while running the graph: {exc}")
        return False
    print("✅ Graph execution completed.")

    # ----------------------------------------------------------------
    section("NODE-BY-NODE EXECUTION LOG")
    print_node_log(final_state.get("node_execution_log", []))

    if final_state.get("errors"):
        section("ERRORS ENCOUNTERED DURING EXECUTION")
        for err in final_state["errors"]:
            print(f"  - {err}")

    if final_state.get("warnings"):
        section("WARNINGS RAISED DURING EXECUTION")
        for w in final_state["warnings"]:
            print(f"  - {w}")

    # ----------------------------------------------------------------
    section("FINAL VALIDATED REPORT SUMMARY")
    report = final_state.get("strategy_report", {})
    validation = final_state.get("validation_results", {})

    if not report:
        print("❌ No strategy report was produced — pipeline did not reach a usable result.")
        return False

    print("\n--- Executive Summary ---")
    print(report.get("executive_summary", "(none)"))

    print("\n--- Section item counts ---")
    for section_name in REQUIRED_SECTIONS[1:]:  # skip executive_summary (not a list)
        items = report.get(section_name, [])
        print(f"  {section_name}: {len(items)} item(s)")

    print("\n--- Validation Results ---")
    print(f"  all_sections_present : {validation.get('all_sections_present')}")
    print(f"  is_valid              : {validation.get('is_valid')}")
    citation_summary = validation.get("citation_summary", {})
    print(f"  total_citations       : {citation_summary.get('total_citations')}")
    print(f"  cited_sources         : {citation_summary.get('cited_sources')}")
    if validation.get("unrecognized_citations"):
        print(f"  unrecognized_citations: {validation.get('unrecognized_citations')}")

    # ----------------------------------------------------------------
    section("VERIFICATION RESULT")
    missing_sections = [s for s in REQUIRED_SECTIONS if s not in report]
    if missing_sections:
        print(f"⚠️  PARTIAL — report generated but missing section(s): {missing_sections}")
        return False
    if not validation.get("is_valid", False):
        print("⚠️  PARTIAL — report generated but the Validator Agent marked it invalid. See warnings above.")
        return False

    print("✅ PASSED — graph executed end-to-end and produced a structurally valid, validated strategy report.")
    return True


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
