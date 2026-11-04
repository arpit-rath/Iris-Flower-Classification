"""Final model evaluation, interpretation, and persistence helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as calculate_confusion_matrix,
)
from sklearn.pipeline import Pipeline

from iris_classifier.data import EXPECTED_SPECIES, FEATURE_COLUMNS


@dataclass(frozen=True)
class EvaluationResult:
    """Measured results from one held-out test-set evaluation."""

    accuracy: float
    predictions: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: pd.DataFrame


def evaluate_model(
    estimator: BaseEstimator,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    target_names: Sequence[str] = EXPECTED_SPECIES,
) -> EvaluationResult:
    """Evaluate a fitted classifier on the held-out test partition."""
    predictions = estimator.predict(features_test)
    labels = list(range(len(target_names)))
    report = classification_report(
        target_test,
        predictions,
        labels=labels,
        target_names=list(target_names),
        output_dict=True,
        zero_division=0,
    )

    return EvaluationResult(
        accuracy=float(accuracy_score(target_test, predictions)),
        predictions=np.asarray(predictions),
        confusion_matrix=calculate_confusion_matrix(
            target_test,
            predictions,
            labels=labels,
        ),
        classification_report=pd.DataFrame(report).transpose(),
    )


def create_confusion_matrix_plot(
    result: EvaluationResult,
    output_path: str | Path,
    target_names: Sequence[str] = EXPECTED_SPECIES,
) -> Path:
    """Create and save an annotated confusion matrix."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        result.confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=list(target_names),
        yticklabels=list(target_names),
        linewidths=0.5,
        ax=axis,
    )
    axis.set_xlabel("Predicted species")
    axis.set_ylabel("Actual species")
    axis.set_title("Held-Out Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def extract_standardized_coefficients(
    estimator: BaseEstimator,
    feature_names: Sequence[str] = FEATURE_COLUMNS,
    target_names: Sequence[str] = EXPECTED_SPECIES,
) -> pd.DataFrame:
    """Return multiclass coefficients from the fitted scaled linear model."""
    classifier = (
        estimator.named_steps.get("classifier")
        if isinstance(estimator, Pipeline)
        else estimator
    )
    if classifier is None or not hasattr(classifier, "coef_"):
        raise ValueError("The selected estimator does not expose coefficients.")

    coefficients = np.asarray(classifier.coef_)
    if coefficients.shape != (len(target_names), len(feature_names)):
        raise ValueError(
            "Coefficient dimensions do not match the species and feature names."
        )

    records = []
    for class_index, species in enumerate(target_names):
        for feature_index, feature in enumerate(feature_names):
            coefficient = float(coefficients[class_index, feature_index])
            records.append(
                {
                    "species": species,
                    "feature": feature,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                }
            )
    return pd.DataFrame(records)


def create_coefficient_heatmap(
    coefficients: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Visualize standardized Logistic Regression coefficients by species."""
    required_columns = {"species", "feature", "coefficient"}
    missing_columns = required_columns.difference(coefficients.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing coefficient columns: {missing}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = coefficients.pivot(
        index="species",
        columns="feature",
        values="coefficient",
    ).loc[list(EXPECTED_SPECIES), list(FEATURE_COLUMNS)]

    figure, axis = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Standardized coefficient"},
        ax=axis,
    )
    axis.set_xlabel("Flower measurement")
    axis.set_ylabel("Species")
    axis.set_title("Logistic Regression Coefficients")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_model_bundle(
    estimator: BaseEstimator,
    output_path: str | Path,
    feature_names: Sequence[str] = FEATURE_COLUMNS,
    target_names: Sequence[str] = EXPECTED_SPECIES,
) -> Path:
    """Save the fitted pipeline together with its input and output labels."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": estimator,
        "feature_names": list(feature_names),
        "target_names": list(target_names),
    }
    joblib.dump(bundle, destination)
    return destination


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load and validate a saved Iris model bundle."""
    bundle = joblib.load(model_path)
    required_keys = {"model", "feature_names", "target_names"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("The saved model bundle has an invalid structure.")
    return bundle


def predict_species(
    bundle: dict[str, Any],
    measurements: Sequence[float],
) -> str:
    """Predict a species from four measurements in documented feature order."""
    feature_names = bundle["feature_names"]
    if len(measurements) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} measurements.")

    sample = pd.DataFrame([list(measurements)], columns=feature_names)
    class_index = int(bundle["model"].predict(sample)[0])
    return str(bundle["target_names"][class_index])
