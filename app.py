import streamlit as st

from src.data_loader import (
    clean_customer_data,
    load_sample_data,
    load_uploaded_data,
    standardize_column_names,
    summarize_data_quality,
    validate_required_columns,
)
from src.insights import (
    add_retention_recommendations,
    calculate_kpis,
    create_business_insights,
)
from src.model_training import save_model_bundle, train_and_select_model
from src.prediction import predict_churn_probabilities
from src.visualizations import (
    churn_distribution_chart,
    confusion_matrix_chart,
    feature_importance_chart,
    monthly_charges_vs_risk_chart,
    risk_by_contract_chart,
    risk_segmentation_chart,
    tenure_vs_probability_chart,
)


st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] {
        color: #4b5563;
    }
    .business-note {
        border-left: 4px solid #2563eb;
        background: #eff6ff;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        color: #1f2937;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    st.title("Customer Churn Prediction Dashboard")
    st.markdown(
        """
        <div class="business-note">
        Businesses lose revenue when customers stop using their product or service.
        This dashboard identifies customers with high churn risk, explains the likely
        drivers, and recommends practical retention actions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_data, data_source_label = load_dashboard_data()
    if raw_data is None:
        st.stop()

    standardized_data = standardize_column_names(raw_data)
    missing_columns = validate_required_columns(standardized_data)
    if missing_columns:
        st.error(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
            + ". Please upload a file with the expected customer churn schema."
        )
        st.stop()

    cleaned_data = clean_customer_data(standardized_data)
    quality_summary = summarize_data_quality(cleaned_data)
    show_sidebar_quality_summary(quality_summary, data_source_label)

    try:
        with st.spinner("Training churn models and scoring customers..."):
            training_result = train_and_select_model(cleaned_data)
            predictions = predict_churn_probabilities(
                training_result["best_model"],
                cleaned_data,
                training_result["feature_columns"],
            )
            predictions = add_retention_recommendations(predictions)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    kpis = calculate_kpis(predictions)
    show_kpi_cards(kpis)

    dashboard_tab, model_tab, customers_tab, insights_tab = st.tabs(
        [
            "Business Dashboard",
            "Model Performance",
            "Customer Predictions",
            "Recommendations & Insights",
        ]
    )

    with dashboard_tab:
        show_business_dashboard(predictions)

    with model_tab:
        show_model_performance(training_result)

    with customers_tab:
        show_customer_prediction_table(predictions)

    with insights_tab:
        show_recommendations_and_insights(predictions, training_result["best_model_name"])


def load_dashboard_data():
    with st.sidebar:
        st.header("Dataset")
        data_option = st.radio(
            "Choose data source",
            ["Use sample dataset", "Upload CSV"],
            label_visibility="collapsed",
        )
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if data_option == "Upload CSV":
        if uploaded_file is None:
            st.info("Upload a CSV file or switch back to the sample dataset.")
            return None, "No file selected"
        try:
            return load_uploaded_data(uploaded_file), uploaded_file.name
        except Exception as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            return None, "Upload error"

    try:
        return load_sample_data(), "Sample dataset"
    except Exception as error:
        st.error(f"Could not load the sample dataset: {error}")
        return None, "Sample dataset"


def show_sidebar_quality_summary(quality_summary: dict, data_source_label: str) -> None:
    with st.sidebar:
        st.divider()
        st.subheader("Data Quality")
        st.write(f"Source: {data_source_label}")
        st.write(f"Rows: {quality_summary['rows']:,}")
        st.write(f"Columns: {quality_summary['columns']:,}")
        st.write(f"Missing values after cleaning: {quality_summary['missing_values']:,}")
        st.write(f"Duplicate customer IDs: {quality_summary['duplicate_customer_ids']:,}")


def show_kpi_cards(kpis: dict) -> None:
    st.subheader("Executive KPI Summary")
    columns = st.columns(5)
    columns[0].metric("Total Customers", f"{kpis['total_customers']:,}")
    columns[1].metric("Predicted Churn Rate", f"{kpis['predicted_churn_rate']:.1%}")
    columns[2].metric("High-Risk Customers", f"{kpis['high_risk_customers']:,}")
    columns[3].metric("Avg Monthly Charges", f"${kpis['average_monthly_charges']:,.2f}")
    columns[4].metric("Revenue at Risk", f"${kpis['annual_revenue_at_risk']:,.0f}")
    st.caption("Revenue at risk is annualized from monthly charges multiplied by churn probability.")


def show_business_dashboard(predictions) -> None:
    st.subheader("Business Dashboard")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(churn_distribution_chart(predictions), use_container_width=True)
    with right:
        st.plotly_chart(risk_segmentation_chart(predictions), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(risk_by_contract_chart(predictions), use_container_width=True)
    with right:
        st.plotly_chart(monthly_charges_vs_risk_chart(predictions), use_container_width=True)

    st.plotly_chart(tenure_vs_probability_chart(predictions), use_container_width=True)


def show_model_performance(training_result: dict) -> None:
    st.subheader("Model Performance")
    st.success(f"Selected model: {training_result['best_model_name']}")
    st.caption(
        f"Training rows: {training_result['training_rows']:,} | "
        f"Testing rows: {training_result['testing_rows']:,}"
    )

    metrics = training_result["best_metrics"]
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    columns[1].metric("Precision", f"{metrics['precision']:.1%}")
    columns[2].metric("Recall", f"{metrics['recall']:.1%}")
    columns[3].metric("F1-score", f"{metrics['f1_score']:.1%}")

    comparison = training_result["model_comparison"].copy()
    for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        comparison[column] = comparison[column].map(lambda value: f"{value:.1%}")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            confusion_matrix_chart(training_result["confusion_matrix"]),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            feature_importance_chart(training_result["feature_importance"]),
            use_container_width=True,
        )

    if st.button("Save selected model with Joblib"):
        saved_path = save_model_bundle(training_result)
        st.success(f"Model saved to {saved_path}")


def show_customer_prediction_table(predictions) -> None:
    st.subheader("Customer-Level Churn Prediction Table")
    risk_options = ["Low Risk", "Medium Risk", "High Risk"]
    selected_risks = st.multiselect("Risk level", risk_options, default=risk_options)

    filtered_predictions = predictions[predictions["risk_level"].isin(selected_risks)].copy()
    filtered_predictions = filtered_predictions.sort_values(
        "churn_probability",
        ascending=False,
    )

    display_columns = [
        "customer_id",
        "gender",
        "contract_type",
        "tenure_months",
        "monthly_charges",
        "support_tickets",
        "last_login_days",
        "churn_probability_pct",
        "predicted_churn",
        "risk_level",
        "retention_recommendation",
    ]
    st.dataframe(
        filtered_predictions[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability_pct": st.column_config.NumberColumn(
                "Churn Probability",
                format="%.1f%%",
            ),
            "monthly_charges": st.column_config.NumberColumn(
                "Monthly Charges",
                format="$%.2f",
            ),
        },
    )

    csv_data = filtered_predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored customers",
        data=csv_data,
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )


def show_recommendations_and_insights(predictions, selected_model_name: str) -> None:
    st.subheader("Retention Recommendations")
    high_risk_customers = (
        predictions[predictions["risk_level"] == "High Risk"]
        .sort_values("churn_probability", ascending=False)
        .head(15)
    )

    recommendation_columns = [
        "customer_id",
        "contract_type",
        "monthly_charges",
        "support_tickets",
        "last_login_days",
        "churn_probability_pct",
        "retention_recommendation",
    ]
    st.dataframe(
        high_risk_customers[recommendation_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability_pct": st.column_config.NumberColumn(
                "Churn Probability",
                format="%.1f%%",
            ),
            "monthly_charges": st.column_config.NumberColumn(
                "Monthly Charges",
                format="$%.2f",
            ),
        },
    )

    st.subheader("Business Insights Summary")
    for insight in create_business_insights(predictions, selected_model_name):
        st.write(f"- {insight}")


if __name__ == "__main__":
    main()
