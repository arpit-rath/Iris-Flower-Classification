"""Model preparation, comparison, and training helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from iris_classifier.data import FEATURE_COLUMNS, TARGET_COLUMN, validate_iris_dataframe

RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_TIE_BREAK_PRIORITY = (
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree",
)


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible, stratified training and test partitions."""
    validate_iris_dataframe(frame)
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_model_candidates(
    random_state: int = RANDOM_STATE,
) -> dict[str, BaseEstimator]:
    """Return the three classification candidates used in the comparison."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


def compare_models(
    candidates: dict[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare candidates through stratified cross-validation."""
    cross_validator = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    records: list[dict[str, float | str]] = []

    for name, estimator in candidates.items():
        scores = cross_validate(
            estimator,
            features,
            target,
            cv=cross_validator,
            scoring="accuracy",
            return_train_score=True,
        )
        records.append(
            {
                "model": name,
                "mean_training_accuracy": scores["train_score"].mean(),
                "mean_validation_accuracy": scores["test_score"].mean(),
                "validation_std": scores["test_score"].std(),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by=["mean_validation_accuracy", "validation_std"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def select_best_model_name(results: pd.DataFrame) -> str:
    """Select by validation accuracy with a deterministic simplicity tie-break."""
    required_columns = {"model", "mean_validation_accuracy"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing comparison columns: {missing}")
    if results.empty:
        raise ValueError("Model comparison results are empty.")

    best_score = results["mean_validation_accuracy"].max()
    tied_models = set(
        results.loc[
            np.isclose(results["mean_validation_accuracy"], best_score),
            "model",
        ]
    )

    for model_name in MODEL_TIE_BREAK_PRIORITY:
        if model_name in tied_models:
            return model_name

    return sorted(str(model_name) for model_name in tied_models)[0]


def train_selected_model(
    model_name: str,
    candidates: dict[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
) -> BaseEstimator:
    """Fit a fresh copy of the selected candidate on all training data."""
    if model_name not in candidates:
        raise ValueError(f"Unknown model: {model_name}")

    estimator = clone(candidates[model_name])
    estimator.fit(features, target)
    return estimator


def create_model_comparison_plot(
    results: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create a cross-validation accuracy comparison with error bars."""
    required_columns = {
        "model",
        "mean_validation_accuracy",
        "validation_std",
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing comparison columns: {missing}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#2A9D8F", "#E9C46A", "#E76F51"]

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(
        results["model"],
        results["mean_validation_accuracy"],
        yerr=results["validation_std"],
        capsize=6,
        color=colors[: len(results)],
        edgecolor="white",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean cross-validation accuracy")
    axis.set_xlabel("Classification model")
    axis.set_title("Training-Set Model Comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination
