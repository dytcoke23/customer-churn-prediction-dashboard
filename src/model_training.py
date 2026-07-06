from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    clean_feature_name,
    create_preprocessor,
    split_features_and_target,
)


def train_and_select_model(dataframe: pd.DataFrame, random_state: int = 42) -> dict:
    """Train two churn models and select the best one by F1-score."""
    X_train, X_test, y_train, y_test = split_features_and_target(
        dataframe=dataframe,
        random_state=random_state,
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    trained_models = {}
    comparison_rows = []

    for model_name, classifier in candidate_models.items():
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", classifier),
            ]
        )
        model_pipeline.fit(X_train, y_train)

        predicted_labels = model_pipeline.predict(X_test)
        predicted_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
        metrics = _calculate_metrics(y_test, predicted_labels, predicted_probabilities)

        trained_models[model_name] = {
            "model": model_pipeline,
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, predicted_labels, labels=[0, 1]),
        }
        comparison_rows.append({"model_name": model_name, **metrics})

    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["f1_score", "recall", "accuracy"],
        ascending=False,
    )
    best_model_name = model_comparison.iloc[0]["model_name"]
    best_model_result = trained_models[best_model_name]

    return {
        "best_model_name": best_model_name,
        "best_model": best_model_result["model"],
        "best_metrics": best_model_result["metrics"],
        "confusion_matrix": best_model_result["confusion_matrix"],
        "model_comparison": model_comparison.reset_index(drop=True),
        "feature_importance": get_feature_importance(best_model_result["model"]),
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
    }


def save_model_bundle(training_result: dict, output_path: str | Path = "models/best_churn_model.joblib") -> Path:
    """Save the selected model and useful metadata for reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_name": training_result["best_model_name"],
        "model": training_result["best_model"],
        "feature_columns": training_result["feature_columns"],
        "metrics": training_result["best_metrics"],
    }
    joblib.dump(bundle, output_path)
    return output_path


def get_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature importance from the selected model."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]
    transformed_feature_names = preprocessor.get_feature_names_out()
    readable_feature_names = [clean_feature_name(name) for name in transformed_feature_names]

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
        importance_type = "Gini importance"
    elif hasattr(classifier, "coef_"):
        importance_values = np.abs(classifier.coef_[0])
        importance_type = "Absolute coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    importance = pd.DataFrame(
        {
            "feature": readable_feature_names,
            "importance": importance_values,
            "importance_type": importance_type,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def _calculate_metrics(y_true, y_pred, y_probability) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probability)
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics
