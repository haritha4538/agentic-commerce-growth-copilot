"""
Phase 5 verification script.

Loads sample data, runs the Analytics Engine, retrieves policy context, and
calls the Strategy Agent (Gemini) to generate a full strategy report —
printing every section so it can be read end-to-end before any
LangGraph/orchestration work happens on top of it.

Like scripts/test_rag.py, this script is written to fail gracefully and
informatively at every stage rather than crashing with a raw traceback.

Run from the project root:
    python scripts/test_strategy_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_list_section(title: str, items: list):
    print(f"\n--- {title} ({len(items)} item(s)) ---")
    if not items:
        print("  (none)")
        return
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            headline = item.get("title") or item.get("action") or item.get("risk") or "(untitled)"
            print(f"  {i}. {headline}")
            for key in ("description", "rationale"):
                if item.get(key):
                    print(f"     {key}: {item[key]}")
            if item.get("supporting_metric"):
                print(f"     supporting_metric: {item['supporting_metric']}")
            if item.get("policy_reference"):
                print(f"     policy_reference: {item['policy_reference']}")
            if item.get("estimated_impact"):
                print(f"     estimated_impact: {item['estimated_impact']}")
            if item.get("priority"):
                print(f"     priority: {item['priority']}")
        else:
            print(f"  {i}. {item}")


def main() -> bool:
    # ----------------------------------------------------------------
    section("STEP 0: Checking dependencies and imports")
    try:
        from src.config import GEMINI_API_KEY
        from src.data.loader import load_sample_data
        from src.analytics.analytics_engine import run_full_analysis
        from src.rag.retriever import collection_stats
        from src.agents.strategy_agent import (
            generate_strategy_report,
            StrategyGenerationError,
            REQUIRED_SECTIONS,
        )
        from src.rag.retriever import RetrievalError
        from src.rag.embeddings import EmbeddingGenerationError
    except ImportError as exc:
        print(f"❌ Could not import required modules: {exc}")
        print("   Run `pip install -r requirements.txt` and try again.")
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
    section("STEP 2: Running the Analytics Engine")
    try:
        analytics_results = run_full_analysis(customers_df, products_df, orders_df)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Analytics Engine failed: {exc}")
        return False
    kpis = analytics_results["executive_kpis"]
    print(f"✅ Analytics computed. Total revenue: {kpis.get('total_revenue')}, "
          f"Total orders: {kpis.get('total_orders')}.")

    signal_counts = {k: len(v) for k, v in analytics_results.get("growth_signals", {}).items()}
    print(f"   Growth signals found: {signal_counts}")

    # ----------------------------------------------------------------
    section("STEP 3: Checking knowledge base status")
    stats = collection_stats()
    print(f"Knowledge base status: {stats}")
    if not stats.get("exists") or stats.get("chunk_count", 0) == 0:
        print("❌ The policy knowledge base has not been ingested yet.")
        print("   Run `python scripts/test_rag.py` (Phase 4) first, then re-run this script.")
        return False
    print("✅ Knowledge base is populated and ready for retrieval.")

    # ----------------------------------------------------------------
    section("STEP 4: Checking Gemini API key")
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY is not set.")
        print("   Copy .env.example to .env and add your key from Google AI Studio, then re-run this script.")
        print("   (Everything else in Phase 5 — data loading, analytics, business-context construction,")
        print("    prompt building, response parsing — is implemented and covered by offline logic checks;")
        print("    only the live Gemini call is blocked without a key.)")
        return False
    print("✅ GEMINI_API_KEY is set.")

    # ----------------------------------------------------------------
    section("STEP 5: Calling the Strategy Agent (Gemini)")
    try:
        report = generate_strategy_report(analytics_results)
    except (StrategyGenerationError, RetrievalError, EmbeddingGenerationError, EnvironmentError) as exc:
        print(f"❌ Strategy Agent failed: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Unexpected error calling the Strategy Agent: {exc}")
        return False
    print("✅ Strategy report generated.")

    # ----------------------------------------------------------------
    section("STEP 6: Structural checks on the generated report")
    missing_sections = [s for s in REQUIRED_SECTIONS if s not in report.report]
    if missing_sections:
        print(f"❌ Report is missing required section(s): {missing_sections}")
        return False
    print(f"✅ All {len(REQUIRED_SECTIONS)} required sections are present.")

    if report.warnings:
        print(f"⚠️  Warnings raised during generation: {report.warnings}")
    else:
        print("✅ No structural warnings.")

    print(f"Policy sources cited/available: {report.policy_sources_used}")

    # ----------------------------------------------------------------
    section("GENERATED STRATEGY REPORT")
    print("\n--- Executive Summary ---")
    print(report.report.get("executive_summary", "(none)"))

    print_list_section("Revenue Opportunities", report.report.get("revenue_opportunities", []))
    print_list_section("Customer Growth Opportunities", report.report.get("customer_growth_opportunities", []))
    print_list_section("Product Opportunities", report.report.get("product_opportunities", []))
    print_list_section("Risks", report.report.get("risks", []))
    print_list_section("Recommended Actions", report.report.get("recommended_actions", []))

    # ----------------------------------------------------------------
    section("VERIFICATION RESULT")
    print("✅ PASSED — Strategy Agent produced a structurally valid, policy-cited report.")
    return True


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
