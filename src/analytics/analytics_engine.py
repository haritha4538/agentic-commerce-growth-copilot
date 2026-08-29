"""
Analytics Engine — Phase 2

Pure, deterministic pandas computations over validated customers/products/
orders DataFrames (see src/data/loader.py and src/data/validator.py).

HARD RULE (do not violate in later phases): every number that ends up in
front of a user or an LLM prompt must trace back to a function in this file.
Agents interpret and prioritize these numbers — they never compute them.

All public functions:
  - accept already-validated DataFrames (as produced by src.data.loader)
  - never raise on empty/missing input — they return empty-but-well-formed
    results instead, so callers (Streamlit, LangGraph nodes) don't need to
    special-case "no data yet"
  - are side-effect free (no I/O, no printing, no global state)

--------------------------------------------------------------------------
KEY ASSUMPTIONS (documented once, applied consistently everywhere below):

1. Revenue basis: only orders with order_status == "Completed" count toward
   revenue, units sold, AOV, and any revenue-based ranking. Cancelled and
   Returned orders are excluded from revenue but are still visible via
   order-status breakdowns if ever needed.

2. Revenue formula per order line:
       revenue = unit_price * quantity * (1 - discount_pct / 100)

3. Profit / margin uses the product's static cost_price and selling_price:
       unit_margin_pct = (selling_price - cost_price) / selling_price
   Realized order-level profit uses the *actual* unit_price paid (after
   product-level pricing), not a recomputed price:
       profit = revenue - (cost_price * quantity)

4. "Overall Growth %" compares the trailing 30-day revenue window to the
   prior 30-day window (day 31-60 back), anchored on the latest order date
   in the dataset. This is more robust than calendar-month comparison when
   the most recent month is only partially elapsed in the data. Returns
   None (not zero, not fabricated) when there isn't enough history.

5. RFM scoring uses quartiles (1 = worst, 4 = best) computed on a rank
   basis to avoid errors from duplicate bin edges on small datasets.
   Recency is inverted (fewer days since last order = higher score).

6. "Total Customers" in the executive KPIs means customers with at least
   one Completed order in the given data (i.e. active/paying customers),
   which is what a growth analysis cares about. The full registered
   customer count is returned separately for reference.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

COMPLETED_STATUS = "Completed"

# Empty-input fallbacks are typed so downstream code (Streamlit tables,
# LangGraph state) can rely on consistent columns even with zero data.
_ORDER_LEVEL_COLUMNS = [
    "order_id", "customer_id", "product_id", "order_date", "quantity",
    "unit_price", "discount_pct", "order_status", "product_name",
    "category", "sub_category", "cost_price", "selling_price",
    "revenue", "cost", "profit", "year_month",
]


# ==========================================================================
# Internal helpers
# ==========================================================================

def _empty(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def prepare_order_facts(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds the single enriched order-line table that every other function
    in this module reads from: parses dates, joins product attributes, and
    computes revenue/cost/profit per order line.

    Returns ALL rows (including Cancelled/Returned) with an `order_status`
    column preserved, so callers can filter as needed. Use
    `filter_completed()` to get the revenue-eligible subset.
    """
    if orders_df is None or orders_df.empty:
        return _empty(_ORDER_LEVEL_COLUMNS)

    df = orders_df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    if products_df is not None and not products_df.empty:
        product_cols = ["product_id", "product_name", "category", "sub_category",
                         "cost_price", "selling_price"]
        available_cols = [c for c in product_cols if c in products_df.columns]
        df = df.merge(products_df[available_cols], on="product_id", how="left")
    else:
        for col in ["product_name", "category", "sub_category", "cost_price", "selling_price"]:
            df[col] = np.nan

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0)
    df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce").fillna(0)

    df["revenue"] = df["unit_price"] * df["quantity"] * (1 - df["discount_pct"] / 100)
    df["cost"] = df["cost_price"] * df["quantity"]
    df["profit"] = df["revenue"] - df["cost"]
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    return df


def filter_completed(order_facts: pd.DataFrame) -> pd.DataFrame:
    """Returns only revenue-eligible (Completed) order lines."""
    if order_facts is None or order_facts.empty:
        return order_facts if order_facts is not None else _empty(_ORDER_LEVEL_COLUMNS)
    return order_facts[order_facts["order_status"] == COMPLETED_STATUS]


def _safe_quartile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Assigns a 1-4 quartile score to a numeric series. Uses rank-based
    binning so duplicate values (common in small datasets, e.g. many
    customers with frequency=1) don't break pd.qcut.

    Returns a Series of Int64 scores (1=worst, 4=best), or all-NaN if the
    input is empty or has fewer than 2 distinct values.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype="Int64")

    if series.nunique(dropna=True) < 2:
        # Not enough variation to segment meaningfully; treat everyone as
        # average (score 2) rather than fabricate a spread.
        return pd.Series(2, index=series.index, dtype="Int64")

    ranks = series.rank(method="first")
    try:
        buckets = pd.qcut(ranks, q=4, labels=[1, 2, 3, 4])
    except ValueError:
        # Fewer unique rank values than quartiles requested
        n_bins = max(2, series.nunique(dropna=True))
        buckets = pd.qcut(ranks, q=min(4, n_bins), labels=False, duplicates="drop") + 1

    scores = buckets.astype("Int64")
    if not higher_is_better:
        scores = pd.Series(5, index=scores.index, dtype="Int64") - scores
    return scores


# ==========================================================================
# 1. EXECUTIVE KPIs
# ==========================================================================

def compute_executive_kpis(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
    customers_df: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    """
    Returns the headline numbers for the Executive Overview page.

    Keys:
        total_revenue, total_orders, total_units_sold, average_order_value,
        active_customers, total_registered_customers,
        revenue_growth_pct_30d (None if not enough history),
        cancelled_orders, returned_orders
    """
    facts = prepare_order_facts(orders_df, products_df)
    completed = filter_completed(facts)

    total_registered_customers = (
        int(customers_df["customer_id"].nunique())
        if customers_df is not None and not customers_df.empty
        else 0
    )

    if completed.empty:
        return {
            "total_revenue": 0.0,
            "total_orders": 0,
            "total_units_sold": 0,
            "average_order_value": 0.0,
            "active_customers": 0,
            "total_registered_customers": total_registered_customers,
            "revenue_growth_pct_30d": None,
            "cancelled_orders": int((facts["order_status"] == "Cancelled").sum()) if not facts.empty else 0,
            "returned_orders": int((facts["order_status"] == "Returned").sum()) if not facts.empty else 0,
        }

    total_revenue = float(completed["revenue"].sum())
    total_orders = int(completed["order_id"].nunique())
    total_units_sold = int(completed["quantity"].sum())
    aov = total_revenue / total_orders if total_orders > 0 else 0.0
    active_customers = int(completed["customer_id"].nunique())

    growth_pct = calculate_revenue_growth_30d(completed)

    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_units_sold": total_units_sold,
        "average_order_value": round(aov, 2),
        "active_customers": active_customers,
        "total_registered_customers": total_registered_customers,
        "revenue_growth_pct_30d": round(growth_pct, 2) if growth_pct is not None else None,
        "cancelled_orders": int((facts["order_status"] == "Cancelled").sum()),
        "returned_orders": int((facts["order_status"] == "Returned").sum()),
    }


def calculate_revenue_growth_30d(completed_facts: pd.DataFrame) -> Optional[float]:
    """
    Trailing 30-day revenue vs the prior 30-day window, anchored on the
    latest order date present in the data. Returns None when there isn't
    at least 60 days of order history to compare (rather than a misleading
    0% or a fabricated number).
    """
    if completed_facts is None or completed_facts.empty:
        return None

    max_date = completed_facts["order_date"].max()
    min_date = completed_facts["order_date"].min()
    if (max_date - min_date).days < 60:
        return None

    last_window_start = max_date - pd.Timedelta(days=30)
    prior_window_start = max_date - pd.Timedelta(days=60)

    last_window_revenue = completed_facts.loc[
        completed_facts["order_date"] > last_window_start, "revenue"
    ].sum()
    prior_window_revenue = completed_facts.loc[
        (completed_facts["order_date"] <= last_window_start)
        & (completed_facts["order_date"] > prior_window_start),
        "revenue",
    ].sum()

    if prior_window_revenue <= 0:
        return None

    return float((last_window_revenue - prior_window_revenue) / prior_window_revenue * 100)


# ==========================================================================
# 2. SALES ANALYTICS
# ==========================================================================

def revenue_by_period(order_facts: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    Revenue and order count grouped by time period.
    freq: 'D' for daily, 'M' for monthly (default), 'W' for weekly.
    Returns columns: period, revenue, orders, units_sold
    """
    completed = filter_completed(order_facts)
    if completed.empty:
        return _empty(["period", "revenue", "orders", "units_sold"])

    period = completed["order_date"].dt.to_period(freq)
    grouped = completed.groupby(period).agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        units_sold=("quantity", "sum"),
    ).reset_index()
    grouped["period"] = grouped["order_date"].astype(str)
    grouped = grouped.drop(columns=["order_date"])
    grouped["revenue"] = grouped["revenue"].round(2)
    return grouped[["period", "revenue", "orders", "units_sold"]]


def monthly_growth(order_facts: pd.DataFrame) -> pd.DataFrame:
    """
    Month-over-month revenue growth.
    Returns columns: period, revenue, orders, revenue_growth_pct
    (first row's growth is NaN — there's no prior month to compare to).
    """
    monthly = revenue_by_period(order_facts, freq="M")
    if monthly.empty:
        return _empty(["period", "revenue", "orders", "revenue_growth_pct"])

    monthly = monthly.sort_values("period").reset_index(drop=True)
    monthly["revenue_growth_pct"] = monthly["revenue"].pct_change() * 100
    monthly["revenue_growth_pct"] = monthly["revenue_growth_pct"].round(2)
    return monthly[["period", "revenue", "orders", "revenue_growth_pct"]]


def revenue_by_category(order_facts: pd.DataFrame) -> pd.DataFrame:
    """Returns columns: category, revenue, orders, units_sold, revenue_share_pct"""
    completed = filter_completed(order_facts)
    if completed.empty:
        return _empty(["category", "revenue", "orders", "units_sold", "revenue_share_pct"])

    grouped = completed.groupby("category", dropna=False).agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        units_sold=("quantity", "sum"),
    ).reset_index()

    total_revenue = grouped["revenue"].sum()
    grouped["revenue_share_pct"] = (
        (grouped["revenue"] / total_revenue * 100).round(2) if total_revenue > 0 else 0.0
    )
    grouped["revenue"] = grouped["revenue"].round(2)
    return grouped.sort_values("revenue", ascending=False).reset_index(drop=True)


def revenue_by_channel(order_facts: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue and AOV by customer acquisition channel.
    Returns columns: acquisition_channel, revenue, orders, customers, aov
    """
    completed = filter_completed(order_facts)
    if completed.empty or customers_df is None or customers_df.empty:
        return _empty(["acquisition_channel", "revenue", "orders", "customers", "aov"])

    merged = completed.merge(
        customers_df[["customer_id", "acquisition_channel"]], on="customer_id", how="left"
    )
    grouped = merged.groupby("acquisition_channel", dropna=False).agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
    ).reset_index()
    grouped["aov"] = (grouped["revenue"] / grouped["orders"]).round(2)
    grouped["revenue"] = grouped["revenue"].round(2)
    return grouped.sort_values("aov", ascending=False).reset_index(drop=True)


# ==========================================================================
# 3. PRODUCT ANALYTICS
# ==========================================================================

def product_performance_table(order_facts: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per product (including products with ZERO completed sales, so
    genuinely under-performing/unsold products are visible, not hidden).

    Returns columns: product_id, product_name, category, cost_price,
    selling_price, margin_pct, units_sold, revenue, profit
    """
    if products_df is None or products_df.empty:
        return _empty([
            "product_id", "product_name", "category", "cost_price", "selling_price",
            "margin_pct", "units_sold", "revenue", "profit",
        ])

    completed = filter_completed(order_facts)

    if completed.empty:
        sales = pd.DataFrame(columns=["product_id", "units_sold", "revenue", "profit"])
    else:
        sales = completed.groupby("product_id").agg(
            units_sold=("quantity", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
        ).reset_index()

    base = products_df[["product_id", "product_name", "category", "cost_price", "selling_price"]].copy()
    base["cost_price"] = pd.to_numeric(base["cost_price"], errors="coerce").fillna(0)
    base["selling_price"] = pd.to_numeric(base["selling_price"], errors="coerce").fillna(0)
    base["margin_pct"] = np.where(
        base["selling_price"] > 0,
        (base["selling_price"] - base["cost_price"]) / base["selling_price"] * 100,
        0.0,
    ).round(2)

    table = base.merge(sales, on="product_id", how="left")
    table["units_sold"] = table["units_sold"].fillna(0).astype(int)
    table["revenue"] = table["revenue"].fillna(0).round(2)
    table["profit"] = table["profit"].fillna(0).round(2)

    return table


def top_n_products(product_table: pd.DataFrame, n: int = 10, by: str = "revenue") -> pd.DataFrame:
    """Top-N products by a given metric column (default: revenue)."""
    if product_table is None or product_table.empty or by not in product_table.columns:
        return product_table if product_table is not None else pd.DataFrame()
    return product_table.sort_values(by, ascending=False).head(n).reset_index(drop=True)


def bottom_n_products(product_table: pd.DataFrame, n: int = 10, by: str = "revenue") -> pd.DataFrame:
    """Bottom-N products by a given metric column. Zero-sale products rank at the bottom naturally."""
    if product_table is None or product_table.empty or by not in product_table.columns:
        return product_table if product_table is not None else pd.DataFrame()
    return product_table.sort_values(by, ascending=True).head(n).reset_index(drop=True)


def high_margin_low_volume_products(
    product_table: pd.DataFrame,
    margin_threshold_pct: float = 50.0,
    volume_percentile: float = 0.25,
) -> pd.DataFrame:
    """
    Products with margin_pct above `margin_threshold_pct` AND units_sold at
    or below the given percentile of units sold across all products —
    i.e. profitable products that aren't moving. These are candidates for
    a pricing/marketing push, not automatically "good" or "bad."
    """
    if product_table is None or product_table.empty:
        return product_table if product_table is not None else pd.DataFrame()

    volume_cutoff = product_table["units_sold"].quantile(volume_percentile)
    filtered = product_table[
        (product_table["margin_pct"] >= margin_threshold_pct)
        & (product_table["units_sold"] <= volume_cutoff)
    ]
    return filtered.sort_values(["margin_pct", "units_sold"], ascending=[False, True]).reset_index(drop=True)


# ==========================================================================
# 4. CUSTOMER ANALYTICS + RFM
# ==========================================================================

def customer_performance_table(order_facts: pd.DataFrame) -> pd.DataFrame:
    """
    One row per customer who has at least one Completed order.
    Returns columns: customer_id, revenue, orders, units_sold, aov,
    first_order_date, last_order_date
    """
    completed = filter_completed(order_facts)
    if completed.empty:
        return _empty([
            "customer_id", "revenue", "orders", "units_sold", "aov",
            "first_order_date", "last_order_date",
        ])

    grouped = completed.groupby("customer_id").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        units_sold=("quantity", "sum"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
    ).reset_index()
    grouped["aov"] = (grouped["revenue"] / grouped["orders"]).round(2)
    grouped["revenue"] = grouped["revenue"].round(2)
    return grouped


def compute_rfm_segmentation(
    order_facts: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Classic RFM segmentation on Completed orders.

    Recency  = days since each customer's last completed order
               (relative to `reference_date`, default = latest order date
               in the whole dataset)
    Frequency = number of distinct completed orders
    Monetary  = total completed revenue

    Each dimension gets a 1-4 quartile score (4 = best). Segment labels
    follow a standard, documented rule set — see _label_rfm_segment().

    Returns columns: customer_id, recency_days, frequency, monetary,
    r_score, f_score, m_score, rfm_score, segment
    """
    completed = filter_completed(order_facts)
    columns = [
        "customer_id", "recency_days", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_score", "segment",
    ]
    if completed.empty:
        return _empty(columns)

    if reference_date is None:
        reference_date = completed["order_date"].max()

    grouped = completed.groupby("customer_id").agg(
        last_order_date=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()

    grouped["recency_days"] = (reference_date - grouped["last_order_date"]).dt.days
    grouped["monetary"] = grouped["monetary"].round(2)

    grouped["r_score"] = _safe_quartile_score(grouped["recency_days"], higher_is_better=False)
    grouped["f_score"] = _safe_quartile_score(grouped["frequency"], higher_is_better=True)
    grouped["m_score"] = _safe_quartile_score(grouped["monetary"], higher_is_better=True)
    grouped["rfm_score"] = (
        grouped["r_score"].astype(int) + grouped["f_score"].astype(int) + grouped["m_score"].astype(int)
    )

    grouped["segment"] = grouped.apply(
        lambda row: _label_rfm_segment(row["r_score"], row["f_score"], row["m_score"]), axis=1
    )

    return grouped[[
        "customer_id", "recency_days", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_score", "segment",
    ]]


def _label_rfm_segment(r_score: int, f_score: int, m_score: int) -> str:
    """
    Deterministic rule-based RFM labeling (documented, not a black box):

      Champions        : r>=3 and f>=3 and m>=3   (recent, frequent, big spender)
      Loyal Customers   : f>=3 and m>=3, but not very recent (r<3)
      At Risk           : r<=2 and (f>=3 or m>=3)  -- were valuable, gone quiet
      New Customers     : r>=3 and f<=2             -- recent but low history yet
      Needs Attention   : r==2 and f==2 and m==2    -- solidly average, no red flags
      Hibernating/Lost  : r<=2 and f<=2 and m<=2    -- inactive and low value
      Others            : anything not matched above
    """
    r, f, m = int(r_score), int(f_score), int(m_score)

    if r >= 3 and f >= 3 and m >= 3:
        return "Champions"
    if f >= 3 and m >= 3 and r < 3:
        return "At Risk"
    if f >= 3 and m >= 3:
        return "Loyal Customers"
    if r >= 3 and f <= 2:
        return "New Customers"
    if r <= 2 and f <= 2 and m <= 2:
        return "Hibernating / Lost"
    if r == 2 and f == 2 and m == 2:
        return "Needs Attention"
    return "Others"


def identify_high_value_customers(rfm_table: pd.DataFrame, top_pct: float = 0.15) -> pd.DataFrame:
    """
    Customers in the top `top_pct` of monetary value. Used both as a
    standalone "who matters most" view and as an input to churn-risk
    detection below.
    """
    if rfm_table is None or rfm_table.empty:
        return rfm_table if rfm_table is not None else pd.DataFrame()

    cutoff = rfm_table["monetary"].quantile(1 - top_pct)
    return rfm_table[rfm_table["monetary"] >= cutoff].sort_values(
        "monetary", ascending=False
    ).reset_index(drop=True)


def identify_churn_risk_customers(
    rfm_table: pd.DataFrame,
    recency_days_threshold: int = 60,
) -> pd.DataFrame:
    """
    Customers who were historically valuable or frequent (f_score >= 3 OR
    m_score >= 3) but haven't ordered recently (recency_days above the
    threshold, default 60 days). This is exactly the "At Risk" /
    "Hibernating but was valuable" pattern — high value + gone quiet.
    """
    if rfm_table is None or rfm_table.empty:
        return rfm_table if rfm_table is not None else pd.DataFrame()

    at_risk = rfm_table[
        (rfm_table["recency_days"] > recency_days_threshold)
        & ((rfm_table["f_score"] >= 3) | (rfm_table["m_score"] >= 3))
    ]
    return at_risk.sort_values("monetary", ascending=False).reset_index(drop=True)


# ==========================================================================
# 5. GROWTH SIGNALS
# ==========================================================================

def detect_seasonal_category_spikes(
    order_facts: pd.DataFrame,
    spike_ratio_threshold: float = 1.4,
) -> List[dict]:
    """
    For each category, finds the single month with the highest revenue and
    flags it as a seasonal spike if that month's revenue is at least
    `spike_ratio_threshold`x the category's average monthly revenue.
    """
    completed = filter_completed(order_facts)
    if completed.empty:
        return []

    monthly_cat = completed.groupby(["category", "year_month"]).agg(
        revenue=("revenue", "sum")
    ).reset_index()

    signals = []
    for category, group in monthly_cat.groupby("category"):
        if len(group) < 2:
            continue
        avg_revenue = group["revenue"].mean()
        peak_row = group.loc[group["revenue"].idxmax()]
        ratio = peak_row["revenue"] / avg_revenue if avg_revenue > 0 else 0
        if ratio >= spike_ratio_threshold:
            signals.append({
                "signal_type": "seasonal_category_spike",
                "category": category,
                "peak_month": peak_row["year_month"],
                "peak_month_revenue": round(float(peak_row["revenue"]), 2),
                "category_avg_monthly_revenue": round(float(avg_revenue), 2),
                "spike_ratio": round(float(ratio), 2),
                "description": (
                    f"{category} revenue in {peak_row['year_month']} was "
                    f"{ratio:.1f}x its average monthly revenue — a seasonal spike worth planning inventory/marketing around."
                ),
            })
    return sorted(signals, key=lambda s: s["spike_ratio"], reverse=True)


def detect_high_aov_channels(
    order_facts: pd.DataFrame,
    customers_df: pd.DataFrame,
    lift_threshold_pct: float = 10.0,
) -> List[dict]:
    """
    Flags acquisition channels whose AOV exceeds the overall AOV by at
    least `lift_threshold_pct` percent.
    """
    channel_table = revenue_by_channel(order_facts, customers_df)
    if channel_table.empty:
        return []

    overall_aov = channel_table["revenue"].sum() / channel_table["orders"].sum() \
        if channel_table["orders"].sum() > 0 else 0
    if overall_aov <= 0:
        return []

    signals = []
    for _, row in channel_table.iterrows():
        lift_pct = (row["aov"] - overall_aov) / overall_aov * 100
        if lift_pct >= lift_threshold_pct:
            signals.append({
                "signal_type": "high_aov_channel",
                "channel": row["acquisition_channel"],
                "channel_aov": round(float(row["aov"]), 2),
                "overall_aov": round(float(overall_aov), 2),
                "lift_pct": round(float(lift_pct), 2),
                "customers": int(row["customers"]),
                "description": (
                    f"{row['acquisition_channel']} customers have an AOV of "
                    f"{row['aov']:.0f} vs the overall {overall_aov:.0f} "
                    f"({lift_pct:.0f}% higher) — a candidate channel to scale spend on."
                ),
            })
    return sorted(signals, key=lambda s: s["lift_pct"], reverse=True)


def detect_high_margin_low_volume_signals(
    product_table: pd.DataFrame,
    margin_threshold_pct: float = 50.0,
    volume_percentile: float = 0.25,
    top_n: int = 10,
) -> List[dict]:
    """Wraps high_margin_low_volume_products() into growth-signal dicts."""
    products = high_margin_low_volume_products(product_table, margin_threshold_pct, volume_percentile)
    if products.empty:
        return []

    signals = []
    for _, row in products.head(top_n).iterrows():
        signals.append({
            "signal_type": "high_margin_low_volume_product",
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "margin_pct": float(row["margin_pct"]),
            "units_sold": int(row["units_sold"]),
            "revenue": float(row["revenue"]),
            "description": (
                f"{row['product_name']} carries a {row['margin_pct']:.0f}% margin but sold "
                f"only {int(row['units_sold'])} unit(s) — a strong candidate for a marketing push "
                f"or bundling rather than a price cut."
            ),
        })
    return signals


def detect_churn_risk_signals(
    rfm_table: pd.DataFrame,
    recency_days_threshold: int = 60,
    top_n: int = 15,
) -> List[dict]:
    """Wraps identify_churn_risk_customers() into growth-signal dicts."""
    at_risk = identify_churn_risk_customers(rfm_table, recency_days_threshold)
    if at_risk.empty:
        return []

    signals = []
    for _, row in at_risk.head(top_n).iterrows():
        signals.append({
            "signal_type": "churn_risk_high_value_customer",
            "customer_id": row["customer_id"],
            "recency_days": int(row["recency_days"]),
            "frequency": int(row["frequency"]),
            "monetary": float(row["monetary"]),
            "segment": row["segment"],
            "description": (
                f"Customer {row['customer_id']} generated {row['monetary']:.0f} in revenue across "
                f"{int(row['frequency'])} order(s) but hasn't ordered in {int(row['recency_days'])} days — "
                f"a retention/win-back candidate."
            ),
        })
    return signals


def generate_growth_signals(
    orders_df: pd.DataFrame,
    products_df: pd.DataFrame,
    customers_df: pd.DataFrame,
) -> Dict[str, List[dict]]:
    """
    Runs all four growth-signal detectors and returns them grouped by type.
    This dict is what gets handed to the Growth Opportunity Agent later —
    every field in every signal is a real, traceable calculated value.
    """
    order_facts = prepare_order_facts(orders_df, products_df)
    product_table = product_performance_table(order_facts, products_df)
    rfm_table = compute_rfm_segmentation(order_facts)

    return {
        "seasonal_category_spikes": detect_seasonal_category_spikes(order_facts),
        "high_aov_channels": detect_high_aov_channels(order_facts, customers_df),
        "high_margin_low_volume_products": detect_high_margin_low_volume_signals(product_table),
        "churn_risk_customers": detect_churn_risk_signals(rfm_table),
    }


# ==========================================================================
# CONVENIENCE: full bundle for Streamlit / LangGraph
# ==========================================================================

def run_full_analysis(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> dict:
    """
    Single entry point that computes everything this module offers and
    returns it as one nested dict/DataFrame bundle. Both the Streamlit
    dashboard and the future LangGraph "analytics_engine_node" should call
    this once per data load rather than recomputing pieces ad hoc.
    """
    order_facts = prepare_order_facts(orders_df, products_df)
    product_table = product_performance_table(order_facts, products_df)
    rfm_table = compute_rfm_segmentation(order_facts)

    return {
        "executive_kpis": compute_executive_kpis(orders_df, products_df, customers_df),
        "revenue_by_month": revenue_by_period(order_facts, freq="M"),
        "revenue_by_day": revenue_by_period(order_facts, freq="D"),
        "monthly_growth": monthly_growth(order_facts),
        "revenue_by_category": revenue_by_category(order_facts),
        "revenue_by_channel": revenue_by_channel(order_facts, customers_df),
        "product_table": product_table,
        "top_products": top_n_products(product_table, n=10),
        "bottom_products": bottom_n_products(product_table, n=10),
        "high_margin_low_volume_products": high_margin_low_volume_products(product_table),
        "customer_table": customer_performance_table(order_facts),
        "rfm_table": rfm_table,
        "high_value_customers": identify_high_value_customers(rfm_table),
        "churn_risk_customers": identify_churn_risk_customers(rfm_table),
        "segment_counts": (
            rfm_table["segment"].value_counts().to_dict() if not rfm_table.empty else {}
        ),
        "growth_signals": generate_growth_signals(orders_df, products_df, customers_df),
    }
