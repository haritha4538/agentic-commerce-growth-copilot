import pandas as pd
import pytest

from src.data.validator import validate_dataset


def test_valid_customers_passes():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "signup_date": ["2024-01-01", "2024-02-15"],
        "city": ["Hyderabad", "Pune"],
        "age_group": ["25-34", "35-44"],
        "acquisition_channel": ["Organic", "Referral"],
    })
    result = validate_dataset(df, "customers")
    assert result.is_valid
    assert result.invalid_row_count == 0
    assert len(result.cleaned_df) == 2


def test_missing_required_column_fails():
    df = pd.DataFrame({
        "customer_id": ["C1"],
        "city": ["Hyderabad"],
    })
    result = validate_dataset(df, "customers")
    assert not result.is_valid
    assert "Missing required columns" in result.errors[0]


def test_invalid_numeric_rows_dropped():
    df = pd.DataFrame({
        "order_id": ["O1", "O2", "O3"],
        "customer_id": ["C1", "C1", "C1"],
        "product_id": ["P1", "P1", "P1"],
        "order_date": ["2024-01-01", "2024-01-02", "not-a-date"],
        "quantity": [1, -5, 2],          # -5 violates min_value
        "unit_price": [100.0, 200.0, 150.0],
        "discount_pct": [0, 10, 5],
        "order_status": ["Completed", "Completed", "Completed"],
    })
    result = validate_dataset(df, "orders")
    # Row 2 (qty -5) and Row 3 (bad date) should be dropped; Row 1 remains
    assert result.invalid_row_count == 2
    assert len(result.cleaned_df) == 1
    assert result.cleaned_df.iloc[0]["order_id"] == "O1"


def test_disallowed_status_value_flagged():
    df = pd.DataFrame({
        "order_id": ["O1", "O2"],
        "customer_id": ["C1", "C1"],
        "product_id": ["P1", "P1"],
        "order_date": ["2024-01-01", "2024-01-02"],
        "quantity": [1, 1],
        "unit_price": [100.0, 100.0],
        "discount_pct": [0, 0],
        "order_status": ["Completed", "Shipped"],  # "Shipped" is not allowed
    })
    result = validate_dataset(df, "orders")
    assert result.invalid_row_count == 1
    assert len(result.cleaned_df) == 1


def test_foreign_key_violation_flagged():
    customers_df = pd.DataFrame({
        "customer_id": ["C1"],
        "signup_date": ["2024-01-01"],
        "city": ["Hyderabad"],
        "age_group": ["25-34"],
        "acquisition_channel": ["Organic"],
    })
    orders_df = pd.DataFrame({
        "order_id": ["O1", "O2"],
        "customer_id": ["C1", "C_UNKNOWN"],   # C_UNKNOWN doesn't exist
        "product_id": ["P1", "P1"],
        "order_date": ["2024-01-01", "2024-01-02"],
        "quantity": [1, 1],
        "unit_price": [100.0, 100.0],
        "discount_pct": [0, 0],
        "order_status": ["Completed", "Completed"],
    })
    products_df = pd.DataFrame({
        "product_id": ["P1"],
        "product_name": ["Widget"],
        "category": ["Electronics"],
        "sub_category": ["Accessories"],
        "cost_price": [100.0],
        "selling_price": [150.0],
        "launch_date": ["2023-01-01"],
    })
    result = validate_dataset(
        orders_df, "orders",
        reference_data={"customers": customers_df, "products": products_df},
    )
    assert result.invalid_row_count == 1
    assert len(result.cleaned_df) == 1
    assert result.cleaned_df.iloc[0]["order_id"] == "O1"


def test_duplicate_primary_key_dropped():
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "signup_date": ["2024-01-01", "2024-01-01", "2024-02-01"],
        "city": ["Hyderabad", "Hyderabad", "Pune"],
        "age_group": ["25-34", "25-34", "35-44"],
        "acquisition_channel": ["Organic", "Organic", "Referral"],
    })
    result = validate_dataset(df, "customers")
    assert result.duplicate_row_count == 1
    assert len(result.cleaned_df) == 2


def test_majority_invalid_marks_dataset_invalid():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "signup_date": ["bad", "bad", "bad", "2024-01-01"],
        "city": ["Hyderabad"] * 4,
        "age_group": ["25-34"] * 4,
        "acquisition_channel": ["Organic"] * 4,
    })
    result = validate_dataset(df, "customers")
    assert not result.is_valid
