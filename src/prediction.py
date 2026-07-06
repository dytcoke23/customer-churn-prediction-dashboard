import numpy as np
import pandas as pd


LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65


def predict_churn_probabilities(
    model,
    customer_data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Add churn probability, predicted label, and risk segment to each customer."""
    missing_columns = [column for column in feature_columns if column not in customer_data.columns]
    if missing_columns:
        raise ValueError(f"Cannot predict because columns are missing: {', '.join(missing_columns)}")

    probabilities = model.predict_proba(customer_data[feature_columns])[:, 1]
    predictions = customer_data.copy()
    predictions["churn_probability"] = probabilities
    predictions["churn_probability_pct"] = (probabilities * 100).round(1)
    predictions["predicted_churn"] = np.where(probabilities >= 0.5, "Yes", "No")
    predictions["risk_level"] = [_classify_risk(probability) for probability in probabilities]
    return predictions


def _classify_risk(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    if probability >= LOW_RISK_THRESHOLD:
        return "Medium Risk"
    return "Low Risk"
