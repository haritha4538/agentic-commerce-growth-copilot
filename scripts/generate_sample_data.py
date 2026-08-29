"""
Generates realistic sample data for the Agentic Commerce Growth Copilot.

This is NOT random noise — it deliberately encodes business patterns so that
the Analytics Engine and Agents have real signal to discover:

  1. Seasonal spike: "Apparel" category spikes in Nov-Dec (festive/holiday season).
  2. Churn-risk segment: a group of previously high-value, high-frequency
     customers who have gone quiet in the last 60-90 days.
  3. Hidden margin opportunity: a handful of high-margin products with
     unusually low sales volume (under-marketed / mispriced).
  4. Channel efficiency gap: "Referral" acquisition channel has a small
     customer base but a noticeably higher Average Order Value.
  5. A generally growing but noisy revenue trend over the observation window.

Run:
    python scripts/generate_sample_data.py

Output:
    data/sample/customers.csv
    data/sample/products.csv
    data/sample/orders.csv
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Observation window: 15 months ending "today" (script run date), so the
# most recent 2-3 months can show the churn-risk drop-off and current trend.
END_DATE = datetime(2026, 8, 25)
START_DATE = END_DATE - timedelta(days=15 * 30)

N_CUSTOMERS = 220
N_PRODUCTS = 55
N_ORDERS = 2600

CITIES = ["Hyderabad", "Mumbai", "Bengaluru", "Delhi", "Pune",
          "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Kochi"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Channel mix: Referral is deliberately small but will be biased toward
# higher unit prices / lower discounts later (higher AOV signal).
CHANNELS = ["Organic", "Paid_Ads", "Referral", "Social_Media", "Email"]
CHANNEL_WEIGHTS = [0.32, 0.28, 0.08, 0.20, 0.12]

CATEGORY_MAP = {
    "Electronics": ["Audio", "Wearables", "Accessories", "Smart Home"],
    "Apparel": ["Menswear", "Womenswear", "Footwear", "Winterwear"],
    "Home & Kitchen": ["Cookware", "Storage", "Decor", "Appliances"],
    "Beauty & Personal Care": ["Skincare", "Haircare", "Fragrance"],
    "Sports & Outdoors": ["Fitness", "Outdoor Gear", "Team Sports"],
    "Books & Stationery": ["Fiction", "Non-Fiction", "Office Supplies"],
}
CATEGORIES = list(CATEGORY_MAP.keys())
# Apparel gets the strongest Nov-Dec seasonal multiplier
CATEGORY_SEASONALITY = {
    "Apparel": 2.6,
    "Electronics": 1.5,
    "Home & Kitchen": 1.2,
    "Beauty & Personal Care": 1.1,
    "Sports & Outdoors": 1.0,
    "Books & Stationery": 0.9,
}

# ---------------------------------------------------------------------------
# 1. CUSTOMERS
# ---------------------------------------------------------------------------
customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, N_CUSTOMERS + 1)]
signup_dates = [
    START_DATE + timedelta(days=int(rng.integers(0, (END_DATE - START_DATE).days - 30)))
    for _ in range(N_CUSTOMERS)
]
channels = rng.choice(CHANNELS, size=N_CUSTOMERS, p=CHANNEL_WEIGHTS)

customers_df = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
    "city": rng.choice(CITIES, size=N_CUSTOMERS),
    "age_group": rng.choice(AGE_GROUPS, size=N_CUSTOMERS, p=[0.18, 0.32, 0.26, 0.15, 0.09]),
    "acquisition_channel": channels,
})

# Designate ~8% of customers as "high value churn-risk": they will be given
# many orders early in the window and none in the last ~75 days.
n_churn_risk = int(N_CUSTOMERS * 0.08)
churn_risk_customers = set(rng.choice(customer_ids, size=n_churn_risk, replace=False))

# ---------------------------------------------------------------------------
# 2. PRODUCTS
# ---------------------------------------------------------------------------
product_rows = []
pid = 1
for _ in range(N_PRODUCTS):
    category = rng.choice(CATEGORIES)
    sub_category = rng.choice(CATEGORY_MAP[category])
    cost_price = round(float(rng.uniform(150, 4500)), 2)
    margin_multiplier = rng.uniform(1.3, 2.6)
    selling_price = round(cost_price * margin_multiplier, 2)
    launch_offset_days = int(rng.integers(0, (END_DATE - START_DATE).days - 10))
    product_rows.append({
        "product_id": f"PRD{str(pid).zfill(4)}",
        "product_name": f"{sub_category} {category.split()[0]} #{pid}",
        "category": category,
        "sub_category": sub_category,
        "cost_price": cost_price,
        "selling_price": selling_price,
        "launch_date": (START_DATE + timedelta(days=launch_offset_days)).strftime("%Y-%m-%d"),
    })
    pid += 1

products_df = pd.DataFrame(product_rows)

# Designate ~10% of products as "hidden gems": high margin (>55%) but will
# be deliberately given low order volume later.
products_df["margin_pct"] = (
    (products_df["selling_price"] - products_df["cost_price"]) / products_df["selling_price"]
)
high_margin_products = products_df[products_df["margin_pct"] > 0.55]["product_id"].tolist()
hidden_gem_products = set(
    rng.choice(high_margin_products, size=min(6, len(high_margin_products)), replace=False)
)
products_df = products_df.drop(columns=["margin_pct"])  # not part of the delivered schema

# ---------------------------------------------------------------------------
# 3. ORDERS
# ---------------------------------------------------------------------------
product_lookup = products_df.set_index("product_id").to_dict("index")
customer_channel = dict(zip(customers_df["customer_id"], customers_df["acquisition_channel"]))
customer_signup = dict(zip(customers_df["customer_id"], pd.to_datetime(customers_df["signup_date"])))

order_rows = []
order_id = 1
total_days = (END_DATE - START_DATE).days
churn_cutoff = END_DATE - timedelta(days=75)

# Build a per-day base order volume with mild overall growth + weekly noise
day_weights = []
for d in range(total_days):
    date = START_DATE + timedelta(days=d)
    growth = 1.0 + 0.35 * (d / total_days)          # gentle upward trend
    weekday_boost = 1.15 if date.weekday() >= 5 else 1.0  # weekend bump
    day_weights.append(growth * weekday_boost)
day_weights = np.array(day_weights)
day_weights = day_weights / day_weights.sum()

order_dates_pool = rng.choice(
    [START_DATE + timedelta(days=int(d)) for d in range(total_days)],
    size=N_ORDERS,
    p=day_weights,
)

product_ids_all = products_df["product_id"].tolist()

for od in order_dates_pool:
    od = pd.Timestamp(od)
    seasonal_bump = od.month in (11, 12)

    # Weighted category choice: boost seasonal category in Nov/Dec
    cat_weights = []
    for c in CATEGORIES:
        w = 1.0
        if seasonal_bump:
            w *= CATEGORY_SEASONALITY[c]
        cat_weights.append(w)
    cat_weights = np.array(cat_weights) / sum(cat_weights)
    category = rng.choice(CATEGORIES, p=cat_weights)

    candidate_products = products_df[products_df["category"] == category]["product_id"].tolist()
    # Hidden gem products get picked much less often
    weights = np.array([
        0.15 if p in hidden_gem_products else 1.0 for p in candidate_products
    ])
    weights = weights / weights.sum()
    product_id = rng.choice(candidate_products, p=weights)

    # Pick a customer who already existed by this order date
    eligible_customers = [c for c, s in customer_signup.items() if s <= od]
    if not eligible_customers:
        continue

    # Churn-risk customers should not appear after the churn cutoff
    if od >= churn_cutoff:
        eligible_customers = [c for c in eligible_customers if c not in churn_risk_customers]
        if not eligible_customers:
            continue
    else:
        # Before the cutoff, churn-risk customers order more frequently (they were "high value")
        boosted = []
        for c in eligible_customers:
            boosted.extend([c] * (4 if c in churn_risk_customers else 1))
        eligible_customers = boosted

    customer_id = eligible_customers[int(rng.integers(0, len(eligible_customers)))]
    channel = customer_channel[customer_id]

    base_price = product_lookup[product_id]["selling_price"]

    # Referral channel: lower discount, effectively higher realized AOV
    if channel == "Referral":
        discount_pct = round(float(rng.choice([0, 0, 0, 5, 10], p=[0.5, 0.2, 0.15, 0.1, 0.05])), 1)
        quantity = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
    else:
        discount_pct = round(float(rng.choice([0, 5, 10, 15, 20], p=[0.35, 0.25, 0.2, 0.12, 0.08])), 1)
        quantity = int(rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10]))

    unit_price = round(base_price, 2)

    status = rng.choice(
        ["Completed", "Cancelled", "Returned"], p=[0.90, 0.05, 0.05]
    )

    order_rows.append({
        "order_id": f"ORD{str(order_id).zfill(5)}",
        "customer_id": customer_id,
        "product_id": product_id,
        "order_date": od.strftime("%Y-%m-%d"),
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_pct": discount_pct,
        "order_status": status,
    })
    order_id += 1

orders_df = pd.DataFrame(order_rows)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
customers_df.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
products_df.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
orders_df.to_csv(os.path.join(OUTPUT_DIR, "orders.csv"), index=False)

print(f"customers.csv -> {len(customers_df)} rows")
print(f"products.csv  -> {len(products_df)} rows")
print(f"orders.csv    -> {len(orders_df)} rows")
print(f"Churn-risk customers embedded: {len(churn_risk_customers)}")
print(f"Hidden-gem (high margin, low volume) products embedded: {len(hidden_gem_products)}")
