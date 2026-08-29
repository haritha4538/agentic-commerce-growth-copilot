"""
Loads customers/products/orders data — either the bundled sample data or a
user-uploaded CSV — and runs it through validation before handing it to the
Analytics Engine.

Design choice: loading and validation are always paired. There is no code
path in this project that hands an unvalidated DataFrame to the analytics
or agent layers.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union, IO

import pandas as pd

from src.config import SAMPLE_CUSTOMERS_PATH, SAMPLE_PRODUCTS_PATH, SAMPLE_ORDERS_PATH
from src.data.validator import validate_dataset, ValidationResult

DATASET_ORDER = ["customers", "products", "orders"]  # products/customers must validate before orders (FK checks)


@dataclass
class LoadedData:
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return all(r.is_valid for r in self.validation_results.values())

    def summary(self) -> dict:
        return {name: r.summary() for name, r in self.validation_results.items()}


def _read_csv(source: Union[str, IO]) -> pd.DataFrame:
    """Reads a CSV from a file path or a file-like object (e.g. Streamlit UploadedFile)."""
    return pd.read_csv(source)


def load_dataset(
    source: Union[str, IO],
    dataset_name: str,
    reference_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> tuple[pd.DataFrame, ValidationResult]:
    """
    Loads a single dataset (from path or uploaded file) and validates it.
    Returns (cleaned_dataframe, validation_result).
    """
    raw_df = _read_csv(source)
    result = validate_dataset(raw_df, dataset_name, reference_data=reference_data)
    cleaned = result.cleaned_df if result.cleaned_df is not None else raw_df
    return cleaned, result


def load_sample_data() -> LoadedData:
    """Loads and validates the three bundled sample CSVs, in FK-safe order."""
    sources = {
        "customers": SAMPLE_CUSTOMERS_PATH,
        "products": SAMPLE_PRODUCTS_PATH,
        "orders": SAMPLE_ORDERS_PATH,
    }
    return _load_all(sources)


def load_uploaded_data(
    customers_file: IO,
    products_file: IO,
    orders_file: IO,
) -> LoadedData:
    """
    Loads and validates three user-uploaded files (e.g. from
    st.file_uploader). Same validation path as the sample data — uploads get
    no special treatment or reduced scrutiny.
    """
    sources = {
        "customers": customers_file,
        "products": products_file,
        "orders": orders_file,
    }
    return _load_all(sources)


def _load_all(sources: Dict[str, Union[str, IO]]) -> LoadedData:
    loaded = LoadedData()
    for name in DATASET_ORDER:
        cleaned_df, result = load_dataset(
            sources[name],
            name,
            reference_data=loaded.dataframes,  # only populated for datasets already processed
        )
        loaded.dataframes[name] = cleaned_df
        loaded.validation_results[name] = result
    return loaded
