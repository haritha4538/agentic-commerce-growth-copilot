"""
Validates a raw DataFrame (sample or user-uploaded) against the canonical
schema defined in schema.py.

This is used both for the bundled sample data (sanity check on startup) and
for any CSV a user uploads through the Streamlit "Upload your data" widget.

Nothing here talks to an LLM. It is pure, deterministic, testable pandas
code — validation must never depend on model output.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.data.schema import (
    SCHEMAS,
    REQUIRED_COLUMNS,
    FOREIGN_KEYS,
    PRIMARY_KEYS,
    ColumnSpec,
)


@dataclass
class ValidationResult:
    dataset_name: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    invalid_row_count: int = 0
    duplicate_row_count: int = 0
    total_rows: int = 0
    cleaned_df: Optional[pd.DataFrame] = None

    def summary(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "invalid_rows_dropped": self.invalid_row_count,
            "duplicate_rows_dropped": self.duplicate_row_count,
            "missing_values": self.missing_value_counts,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _coerce_column(series: pd.Series, spec: ColumnSpec) -> tuple[pd.Series, pd.Series]:
    """
    Attempts to coerce a column to its expected dtype.
    Returns (coerced_series, is_invalid_mask) where is_invalid_mask marks
    values that could not be coerced (excluding original NaNs).
    """
    original_na = series.isna()

    if spec.dtype in ("int", "float"):
        coerced = pd.to_numeric(series, errors="coerce")
    elif spec.dtype == "date":
        coerced = pd.to_datetime(series, errors="coerce")
    else:  # string / category
        coerced = series.astype("string")

    newly_invalid = coerced.isna() & ~original_na
    return coerced, newly_invalid


def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    reference_data: Optional[Dict[str, pd.DataFrame]] = None,
    drop_invalid: bool = True,
) -> ValidationResult:
    """
    Validates `df` against the schema registered for `dataset_name`
    ("customers" | "products" | "orders").

    reference_data: other already-loaded datasets, used for foreign-key
    checks (e.g. orders.customer_id must exist in customers.customer_id).
    """
    if dataset_name not in SCHEMAS:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Expected one of {list(SCHEMAS)}")

    result = ValidationResult(dataset_name=dataset_name, is_valid=True, total_rows=len(df))
    schema = SCHEMAS[dataset_name]
    working_df = df.copy()

    # 1. Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS[dataset_name] if c not in working_df.columns]
    if missing_cols:
        result.errors.append(f"Missing required columns: {missing_cols}")
        result.is_valid = False
        return result  # cannot proceed further without required columns

    # Drop unexpected extra columns (warn, don't fail)
    expected_cols = {c.name for c in schema}
    extra_cols = [c for c in working_df.columns if c not in expected_cols]
    if extra_cols:
        result.warnings.append(f"Ignoring unexpected extra columns: {extra_cols}")
        working_df = working_df[[c for c in working_df.columns if c in expected_cols]]

    invalid_mask = pd.Series(False, index=working_df.index)

    # 2. Per-column dtype coercion + missing value tracking
    for spec in schema:
        col = working_df[spec.name]

        missing_count = int(col.isna().sum())
        if missing_count > 0:
            result.missing_value_counts[spec.name] = missing_count
            if spec.required:
                result.warnings.append(
                    f"Column '{spec.name}' has {missing_count} missing value(s)."
                )

        coerced, newly_invalid = _coerce_column(col, spec)
        working_df[spec.name] = coerced
        invalid_mask |= newly_invalid
        if newly_invalid.any():
            result.warnings.append(
                f"Column '{spec.name}': {int(newly_invalid.sum())} value(s) could not be "
                f"parsed as {spec.dtype} and were flagged invalid."
            )

        # Range checks
        if spec.min_value is not None and spec.dtype in ("int", "float"):
            below_min = coerced < spec.min_value
            below_min = below_min.fillna(False)
            if below_min.any():
                invalid_mask |= below_min
                result.warnings.append(
                    f"Column '{spec.name}': {int(below_min.sum())} value(s) below minimum "
                    f"allowed value ({spec.min_value})."
                )

        # Allowed-value checks
        if spec.allowed_values is not None:
            bad_values = ~coerced.isin(spec.allowed_values) & coerced.notna()
            if bad_values.any():
                invalid_mask |= bad_values
                result.warnings.append(
                    f"Column '{spec.name}': {int(bad_values.sum())} value(s) outside "
                    f"allowed set {spec.allowed_values}."
                )

        # Any required column that is still null after coercion is invalid
        if spec.required:
            invalid_mask |= working_df[spec.name].isna()

    result.invalid_row_count = int(invalid_mask.sum())

    # 3. Duplicate primary key check
    pk = PRIMARY_KEYS.get(dataset_name)
    duplicate_mask = pd.Series(False, index=working_df.index)
    if pk and pk in working_df.columns:
        duplicate_mask = working_df.duplicated(subset=[pk], keep="first")
        result.duplicate_row_count = int(duplicate_mask.sum())
        if result.duplicate_row_count > 0:
            result.warnings.append(
                f"{result.duplicate_row_count} duplicate '{pk}' value(s) found; "
                f"keeping first occurrence only."
            )

    # 4. Referential integrity (foreign keys)
    if reference_data and dataset_name in FOREIGN_KEYS:
        for fk_col, ref_dataset in FOREIGN_KEYS[dataset_name].items():
            ref_df = reference_data.get(ref_dataset)
            if ref_df is None or fk_col not in working_df.columns:
                continue
            ref_pk = PRIMARY_KEYS[ref_dataset]
            if ref_pk not in ref_df.columns:
                continue
            valid_ids = set(ref_df[ref_pk].astype("string"))
            fk_invalid = ~working_df[fk_col].astype("string").isin(valid_ids)
            fk_invalid = fk_invalid.fillna(True)
            if fk_invalid.any():
                invalid_mask |= fk_invalid
                result.warnings.append(
                    f"Column '{fk_col}': {int(fk_invalid.sum())} value(s) do not match any "
                    f"'{ref_pk}' in {ref_dataset}."
                )
                result.invalid_row_count = int(invalid_mask.sum())

    # 5. Build cleaned_df
    rows_to_drop = invalid_mask | duplicate_mask
    result.cleaned_df = working_df[~rows_to_drop].reset_index(drop=True) if drop_invalid else working_df

    if len(result.cleaned_df) == 0:
        result.errors.append("No valid rows remained after validation.")
        result.is_valid = False
    elif result.invalid_row_count > 0.5 * result.total_rows:
        result.errors.append(
            f"More than 50% of rows ({result.invalid_row_count}/{result.total_rows}) failed validation."
        )
        result.is_valid = False

    return result
