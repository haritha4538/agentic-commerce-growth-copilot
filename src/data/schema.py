"""
Canonical schema definitions for the three core datasets.

Every other module (loader, validator, analytics engine, upload validator)
imports from here so there is exactly one source of truth for "what a valid
customers/products/orders row looks like."
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str            # one of: "string", "int", "float", "date", "category"
    required: bool = True
    allowed_values: Optional[tuple] = None   # for category-like validation
    min_value: Optional[float] = None        # for numeric sanity checks


# ---------------------------------------------------------------------------
# CUSTOMERS
# ---------------------------------------------------------------------------
CUSTOMERS_SCHEMA = [
    ColumnSpec("customer_id", "string"),
    ColumnSpec("signup_date", "date"),
    ColumnSpec("city", "string"),
    ColumnSpec("age_group", "string"),
    ColumnSpec("acquisition_channel", "string"),
]

# ---------------------------------------------------------------------------
# PRODUCTS
# ---------------------------------------------------------------------------
PRODUCTS_SCHEMA = [
    ColumnSpec("product_id", "string"),
    ColumnSpec("product_name", "string"),
    ColumnSpec("category", "string"),
    ColumnSpec("sub_category", "string"),
    ColumnSpec("cost_price", "float", min_value=0),
    ColumnSpec("selling_price", "float", min_value=0),
    ColumnSpec("launch_date", "date"),
]

# ---------------------------------------------------------------------------
# ORDERS
# ---------------------------------------------------------------------------
ORDERS_SCHEMA = [
    ColumnSpec("order_id", "string"),
    ColumnSpec("customer_id", "string"),
    ColumnSpec("product_id", "string"),
    ColumnSpec("order_date", "date"),
    ColumnSpec("quantity", "int", min_value=1),
    ColumnSpec("unit_price", "float", min_value=0),
    ColumnSpec("discount_pct", "float", min_value=0),
    ColumnSpec(
        "order_status", "string",
        allowed_values=("Completed", "Cancelled", "Returned"),
    ),
]

SCHEMAS = {
    "customers": CUSTOMERS_SCHEMA,
    "products": PRODUCTS_SCHEMA,
    "orders": ORDERS_SCHEMA,
}

REQUIRED_COLUMNS = {
    dataset: [c.name for c in spec if c.required]
    for dataset, spec in SCHEMAS.items()
}

# Foreign-key relationships used for referential-integrity checks
FOREIGN_KEYS = {
    "orders": {
        "customer_id": "customers",
        "product_id": "products",
    }
}

# Primary keys, used for duplicate-row checks
PRIMARY_KEYS = {
    "customers": "customer_id",
    "products": "product_id",
    "orders": "order_id",
}
