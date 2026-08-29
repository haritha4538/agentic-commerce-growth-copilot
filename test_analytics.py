"""
Phase 2 verification script.

Loads the bundled sample data through the existing Phase 1 loader/validator,
runs it through the Analytics Engine, and prints the key outputs so we can
eyeball that everything is real, non-empty, and sane before wiring it into
Streamlit or any agent.

Run from the project root:
    python scripts/test_analytics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_sample_data
from src.analytics.analytics_engine import (
    prepare_order_facts,
    compute_executive_kpis,
    revenue_by_category,
    revenue_by_channel,
    monthly_growth,
    product_performance_table,
    top_n_products,
    bottom_n_products,
    high_margin_low_volume_products,
    compute_rfm_segmentation,
    identify_high_value_customers,
    identify_churn_risk_customers,
    generate_growth_signals,
)


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    loaded = load_sample_data()
    if not loaded.is_valid:
        print("Data failed validation — aborting. Summary:")
        print(loaded.summary())
        return

    customers_df = loaded.dataframes["customers"]
    products_df = loaded.dataframes["products"]
    orders_df = loaded.dataframes["orders"]

    print(f"Loaded: {len(customers_df)} customers, {len(products_df)} products, "
          f"{len(orders_df)} orders (all passed validation).")

    order_facts = prepare_order_facts(orders_df, products_df)

    # ---------------------------------------------------------------
    section("EXECUTIVE KPIs")
    kpis = compute_executive_kpis(orders_df, products_df, customers_df)
    for k, v in kpis.items():
        print(f"  {k:32s}: {v}")

    # ---------------------------------------------------------------
    section("MONTHLY REVENUE + GROWTH (last 6 months)")
    growth_df = monthly_growth(order_facts)
    print(growth_df.tail(6).to_string(index=False))

    # ---------------------------------------------------------------
    section("REVENUE BY CATEGORY")
    cat_df = revenue_by_category(order_facts)
    print(cat_df.to_string(index=False))

    # ---------------------------------------------------------------
    section("REVENUE BY ACQUISITION CHANNEL (sorted by AOV)")
    channel_df = revenue_by_channel(order_facts, customers_df)
    print(channel_df.to_string(index=False))

    # ---------------------------------------------------------------
    section("TOP 5 PRODUCTS BY REVENUE")
    product_table = product_performance_table(order_facts, products_df)
    top5 = top_n_products(product_table, n=5)
    print(top5[["product_id", "product_name", "category", "revenue", "units_sold", "margin_pct"]].to_string(index=False))

    section("BOTTOM 5 PRODUCTS BY REVENUE")
    bottom5 = bottom_n_products(product_table, n=5)
    print(bottom5[["product_id", "product_name", "category", "revenue", "units_sold", "margin_pct"]].to_string(index=False))

    section("HIGH-MARGIN / LOW-VOLUME PRODUCTS (margin >= 50%, bottom quartile volume)")
    hmlv = high_margin_low_volume_products(product_table)
    print(f"Found {len(hmlv)} product(s).")
    if not hmlv.empty:
        print(hmlv[["product_id", "product_name", "margin_pct", "units_sold", "revenue"]].head(10).to_string(index=False))

    # ---------------------------------------------------------------
    section("CUSTOMER SEGMENT COUNTS (RFM)")
    rfm_table = compute_rfm_segmentation(order_facts)
    print(f"Customers with >=1 completed order: {len(rfm_table)}")
    print(rfm_table["segment"].value_counts().to_string())

    section("HIGH-VALUE CUSTOMERS (top 15% by monetary) — sample of 5")
    hv = identify_high_value_customers(rfm_table)
    print(f"Total high-value customers: {len(hv)}")
    print(hv[["customer_id", "recency_days", "frequency", "monetary", "segment"]].head(5).to_string(index=False))

    section("CHURN-RISK CUSTOMERS (recency > 60 days, previously high F or M) — sample of 5")
    churn = identify_churn_risk_customers(rfm_table)
    print(f"Total churn-risk customers: {len(churn)}")
    if not churn.empty:
        print(churn[["customer_id", "recency_days", "frequency", "monetary", "segment"]].head(5).to_string(index=False))

    # ---------------------------------------------------------------
    section("GROWTH SIGNALS")
    signals = generate_growth_signals(orders_df, products_df, customers_df)
    for signal_type, items in signals.items():
        print(f"\n--- {signal_type} ({len(items)} found) ---")
        for item in items[:3]:
            print(f"  - {item['description']}")

    section("DONE")
    print("Analytics engine verification completed successfully.")


if __name__ == "__main__":
    main()
