import pandas as pd


def calculate_kpis(predictions: pd.DataFrame) -> dict:
    """Calculate executive-friendly churn KPIs from customer-level predictions."""
    total_customers = len(predictions)
    predicted_churn_customers = int((predictions["predicted_churn"] == "Yes").sum())
    high_risk_customers = int((predictions["risk_level"] == "High Risk").sum())

    monthly_revenue_at_risk = (
        predictions["monthly_charges"].fillna(0) * predictions["churn_probability"].fillna(0)
    ).sum()

    return {
        "total_customers": total_customers,
        "predicted_churn_customers": predicted_churn_customers,
        "predicted_churn_rate": predicted_churn_customers / total_customers
        if total_customers
        else 0,
        "high_risk_customers": high_risk_customers,
        "average_churn_probability": predictions["churn_probability"].mean()
        if total_customers
        else 0,
        "average_monthly_charges": predictions["monthly_charges"].mean()
        if total_customers
        else 0,
        "monthly_revenue_at_risk": monthly_revenue_at_risk,
        "annual_revenue_at_risk": monthly_revenue_at_risk * 12,
    }


def add_retention_recommendations(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach rule-based retention actions to every customer."""
    enriched = predictions.copy()
    high_charge_threshold = enriched["monthly_charges"].quantile(0.75)
    enriched["retention_recommendation"] = enriched.apply(
        lambda row: _recommend_for_customer(row, high_charge_threshold),
        axis=1,
    )
    return enriched


def create_business_insights(predictions: pd.DataFrame, selected_model_name: str) -> list[str]:
    """Create plain-language insights for business users."""
    if predictions.empty:
        return ["No customer records are available for insight generation."]

    kpis = calculate_kpis(predictions)
    insights = [
        f"{selected_model_name} was selected because it achieved the strongest F1-score on the test set.",
        (
            f"{kpis['predicted_churn_rate']:.1%} of customers are predicted to churn, "
            f"with {kpis['high_risk_customers']} customers in the high-risk segment."
        ),
        (
            f"Estimated annualized revenue at risk is ${kpis['annual_revenue_at_risk']:,.0f}, "
            "based on each customer's monthly charge and churn probability."
        ),
    ]

    contract_summary = (
        predictions.groupby("contract_type")["churn_probability"]
        .mean()
        .sort_values(ascending=False)
    )
    if not contract_summary.empty:
        top_contract = contract_summary.index[0]
        insights.append(
            f"Customers on {top_contract} contracts show the highest average churn risk."
        )

    high_risk = predictions[predictions["risk_level"] == "High Risk"]
    if not high_risk.empty:
        avg_tickets = high_risk["support_tickets"].mean()
        avg_inactive_days = high_risk["last_login_days"].mean()
        insights.append(
            f"High-risk customers average {avg_tickets:.1f} support tickets and "
            f"{avg_inactive_days:.0f} days since last login, so service recovery and reactivation should be prioritized."
        )

    return insights


def _recommend_for_customer(row: pd.Series, high_charge_threshold: float) -> str:
    recommendations = []

    if row.get("risk_level") == "High Risk" and row.get("contract_type") == "Month-to-month":
        recommendations.append("Offer an annual plan discount")

    if row.get("support_tickets", 0) >= 3:
        recommendations.append("Route to priority customer support")

    if row.get("monthly_charges", 0) >= high_charge_threshold:
        recommendations.append("Send a personalized pricing or bundle offer")

    if row.get("tenure_months", 0) <= 6:
        recommendations.append("Add to onboarding email campaign")

    if row.get("last_login_days", 0) >= 30:
        recommendations.append("Launch reactivation campaign")

    if not recommendations:
        recommendations.append("Maintain regular engagement and monitor usage")

    return " | ".join(recommendations[:3])
