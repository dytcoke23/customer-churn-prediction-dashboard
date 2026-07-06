from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_customer_churn.csv"

REQUIRED_COLUMNS = [
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "internet_service",
    "contract_type",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "last_login_days",
    "churn",
]

NUMERIC_COLUMNS = [
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "last_login_days",
]

YES_NO_COLUMNS = [
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
    "churn",
]


def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample dataset used when no CSV is uploaded."""
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(f"Sample dataset was not found at {SAMPLE_DATA_PATH}")
    return pd.read_csv(SAMPLE_DATA_PATH)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Read a CSV file uploaded through Streamlit."""
    if uploaded_file is None:
        raise ValueError("No uploaded file was provided.")
    return pd.read_csv(uploaded_file)


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CSV header variations to snake_case."""
    cleaned = dataframe.copy()
    cleaned.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in cleaned.columns
    ]
    return cleaned


def validate_required_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return any required columns that are missing from the dataframe."""
    return [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]


def clean_customer_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer data while keeping the original business fields visible."""
    cleaned = dataframe.copy()

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    if "customer_id" in cleaned.columns:
        missing_ids = cleaned["customer_id"].isna()
        cleaned.loc[missing_ids, "customer_id"] = [
            f"CUST-MISSING-{index + 1:04d}" for index in range(missing_ids.sum())
        ]

    for column in YES_NO_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = _normalize_yes_no(cleaned[column])

    if "senior_citizen" in cleaned.columns:
        cleaned["senior_citizen"] = _normalize_senior_citizen(cleaned["senior_citizen"])

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = cleaned[column].clip(lower=0)

    if {"total_charges", "monthly_charges", "tenure_months"}.issubset(cleaned.columns):
        missing_total_charges = cleaned["total_charges"].isna()
        cleaned.loc[missing_total_charges, "total_charges"] = (
            cleaned.loc[missing_total_charges, "monthly_charges"]
            * cleaned.loc[missing_total_charges, "tenure_months"]
        )

    categorical_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in NUMERIC_COLUMNS and column in cleaned.columns
    ]
    for column in categorical_columns:
        if column != "churn":
            cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned


def summarize_data_quality(dataframe: pd.DataFrame) -> dict[str, int]:
    """Create simple data quality metrics for the sidebar."""
    return {
        "rows": int(len(dataframe)),
        "columns": int(dataframe.shape[1]),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_customer_ids": int(dataframe["customer_id"].duplicated().sum())
        if "customer_id" in dataframe.columns
        else 0,
    }


def _normalize_yes_no(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    mapping = {
        "yes": "Yes",
        "y": "Yes",
        "true": "Yes",
        "1": "Yes",
        "1.0": "Yes",
        "no": "No",
        "n": "No",
        "false": "No",
        "0": "No",
        "0.0": "No",
    }
    return text.map(mapping)


def _normalize_senior_citizen(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "1.0": 1,
        "senior": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "0.0": 0,
        "non-senior": 0,
    }
    return text.map(mapping)
