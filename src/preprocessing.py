import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "last_login_days",
]

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "internet_service",
    "contract_type",
    "paperless_billing",
    "payment_method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Build preprocessing steps fitted only on training data to avoid leakage."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_features_and_target(
    dataframe: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Separate features from the churn label and create a stratified split."""
    missing_features = [column for column in FEATURE_COLUMNS if column not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError("The dataset must include a churn column with Yes/No labels.")

    encoded_target = dataframe[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    usable_rows = encoded_target.notna()
    model_data = dataframe.loc[usable_rows].copy()
    target = encoded_target.loc[usable_rows].astype(int)

    if target.nunique() < 2:
        raise ValueError("The churn column must contain both Yes and No values to train a model.")

    features = model_data[FEATURE_COLUMNS]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def clean_feature_name(feature_name: str) -> str:
    """Make transformed feature names readable for a dashboard chart."""
    clean_name = feature_name
    for prefix in ("numeric__", "categorical__"):
        clean_name = clean_name.replace(prefix, "")

    for column in CATEGORICAL_FEATURES:
        category_prefix = f"{column}_"
        if clean_name.startswith(category_prefix):
            category = clean_name.replace(category_prefix, "", 1)
            return f"{column.replace('_', ' ').title()}: {category}"

    return clean_name.replace("_", " ").title()
