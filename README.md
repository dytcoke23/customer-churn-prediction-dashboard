# Customer Churn Prediction Dashboard

An end-to-end machine learning dashboard that predicts customer churn risk and turns model output into practical retention insights.

## Overview

Customer churn is a business problem where customers stop using a product or service. This Streamlit dashboard helps teams identify customers who are likely to churn, estimate revenue at risk, understand churn drivers, and choose retention actions.

The project is designed as a GitHub-ready portfolio project for entry-level Data Analyst, Business Analyst, AI/ML, and AI Automation roles.

## Business Problem

Businesses lose recurring revenue when customers cancel, become inactive, or move to competitors. A churn prediction workflow helps the business act earlier by finding customers with warning signs such as short tenure, month-to-month contracts, high support tickets, high monthly charges, or long inactivity periods.

## Why Churn Prediction Matters

- Helps retention teams prioritize the customers most likely to leave.
- Connects machine learning predictions to revenue impact.
- Explains which customer behaviors and plan attributes drive churn risk.
- Supports targeted retention campaigns instead of generic discounts.

## Features

- Loads a bundled sample customer churn dataset by default.
- Supports user CSV upload with required-column validation.
- Cleans missing values and standardizes common data formats.
- Trains Logistic Regression and Random Forest models.
- Selects the better model automatically using F1-score.
- Predicts churn probability for every customer.
- Segments customers into Low Risk, Medium Risk, and High Risk.
- Shows accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Calculates business KPIs including revenue at risk.
- Includes interactive Plotly charts for churn distribution, contract risk, monthly charges, tenure, and feature importance.
- Provides a customer-level prediction table with CSV download.
- Generates rule-based retention recommendations.
- Summarizes business insights in simple language.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

## Dataset

The project includes `data/sample_customer_churn.csv` with 300 realistic customer records.

Columns:

- `customer_id`
- `gender`
- `senior_citizen`
- `partner`
- `dependents`
- `tenure_months`
- `phone_service`
- `internet_service`
- `contract_type`
- `paperless_billing`
- `payment_method`
- `monthly_charges`
- `total_charges`
- `support_tickets`
- `last_login_days`
- `churn`

The sample data is synthetic and created for portfolio, learning, and demonstration purposes.

## Machine Learning Workflow

1. Load sample data or uploaded customer CSV.
2. Validate that all required columns are present.
3. Clean customer records and convert numeric fields.
4. Split data into train and test sets with stratification.
5. Fit preprocessing only on the training data to avoid data leakage.
6. Impute missing values, scale numeric fields, and one-hot encode categorical fields.
7. Train Logistic Regression and Random Forest models.
8. Compare models using F1-score.
9. Score every customer with churn probability and risk level.
10. Generate business KPIs, charts, and recommendations.

## Model Performance

The dashboard calculates model performance each time it runs. It displays:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Model comparison table
- Feature importance chart

F1-score is used for model selection because churn prediction usually needs a balance between finding likely churners and avoiding too many false alarms.

## Screenshots

Add screenshots to the `screenshots/` folder after running the app locally.

Recommended screenshots:

- Main dashboard with KPI cards and churn charts.
- Model performance section with metrics and confusion matrix.
- Feature importance chart.
- Customer prediction table.
- Retention recommendations and business insights summary.

## Folder Structure

```text
customer-churn-prediction-dashboard/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- data/
|   |-- sample_customer_churn.csv
|
|-- models/
|   |-- .gitkeep
|
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- model_training.py
|   |-- prediction.py
|   |-- insights.py
|   |-- visualizations.py
|
|-- screenshots/
    |-- .gitkeep
```

## How to Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

- Add SHAP explanations for individual customer predictions.
- Add campaign simulation for discount cost versus retained revenue.
- Store prediction history in a lightweight database.
- Add filters by customer segment, contract type, and payment method.
- Add automated model retraining from newly uploaded data.
- Add exportable PDF or PowerPoint business reports.


