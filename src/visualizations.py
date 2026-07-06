import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


RISK_COLOR_MAP = {
    "Low Risk": "#16a34a",
    "Medium Risk": "#f59e0b",
    "High Risk": "#dc2626",
}

CHURN_COLOR_MAP = {
    "No": "#2563eb",
    "Yes": "#dc2626",
}


def churn_distribution_chart(predictions: pd.DataFrame) -> go.Figure:
    distribution = (
        predictions["predicted_churn"]
        .value_counts()
        .rename_axis("predicted_churn")
        .reset_index(name="customers")
    )
    fig = px.pie(
        distribution,
        names="predicted_churn",
        values="customers",
        title="Predicted Churn Distribution",
        color="predicted_churn",
        color_discrete_map=CHURN_COLOR_MAP,
        hole=0.45,
    )
    return _apply_layout(fig)


def risk_segmentation_chart(predictions: pd.DataFrame) -> go.Figure:
    order = ["Low Risk", "Medium Risk", "High Risk"]
    segmentation = (
        predictions["risk_level"]
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("risk_level")
        .reset_index(name="customers")
    )
    fig = px.bar(
        segmentation,
        x="risk_level",
        y="customers",
        title="Customer Risk Segmentation",
        color="risk_level",
        color_discrete_map=RISK_COLOR_MAP,
        text="customers",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Customers", xaxis_title="")
    return _apply_layout(fig)


def risk_by_contract_chart(predictions: pd.DataFrame) -> go.Figure:
    summary = (
        predictions.groupby("contract_type")
        .agg(
            avg_churn_probability=("churn_probability", "mean"),
            customers=("customer_id", "count"),
        )
        .reset_index()
        .sort_values("avg_churn_probability", ascending=False)
    )
    summary["risk_label"] = summary["avg_churn_probability"].map(lambda value: f"{value:.1%}")

    fig = px.bar(
        summary,
        x="contract_type",
        y="avg_churn_probability",
        title="Churn Risk by Contract Type",
        color="contract_type",
        text="risk_label",
        hover_data={"customers": True, "avg_churn_probability": ":.1%"},
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Contract Type",
        yaxis_title="Average Churn Probability",
        yaxis_tickformat=".0%",
    )
    fig.update_traces(textposition="outside")
    return _apply_layout(fig)


def monthly_charges_vs_risk_chart(predictions: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        predictions,
        x="monthly_charges",
        y="churn_probability",
        color="risk_level",
        color_discrete_map=RISK_COLOR_MAP,
        title="Monthly Charges vs Churn Risk",
        hover_data=["customer_id", "contract_type", "support_tickets"],
    )
    fig.update_layout(
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Churn Probability",
        yaxis_tickformat=".0%",
    )
    return _apply_layout(fig)


def tenure_vs_probability_chart(predictions: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        predictions,
        x="tenure_months",
        y="churn_probability",
        color="risk_level",
        color_discrete_map=RISK_COLOR_MAP,
        title="Tenure vs Churn Probability",
        hover_data=["customer_id", "contract_type", "monthly_charges"],
    )
    fig.update_layout(
        xaxis_title="Tenure (Months)",
        yaxis_title="Churn Probability",
        yaxis_tickformat=".0%",
    )
    return _apply_layout(fig)


def feature_importance_chart(feature_importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top_features = feature_importance.head(top_n).sort_values("importance", ascending=True)
    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Churn Drivers",
        color="importance",
        color_continuous_scale=["#93c5fd", "#1d4ed8"],
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    return _apply_layout(fig)


def confusion_matrix_chart(confusion_matrix_values) -> go.Figure:
    labels = ["No Churn", "Churn"]
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix_values,
            x=[f"Predicted {label}" for label in labels],
            y=[f"Actual {label}" for label in labels],
            colorscale="Blues",
            text=confusion_matrix_values,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Customers: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
    )
    return _apply_layout(fig)


def _apply_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=30, t=70, b=40),
        font=dict(family="Arial", size=13, color="#111827"),
        title_font=dict(size=18, color="#111827"),
        legend_title_text="",
    )
    return fig
