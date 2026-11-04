"""Tests for model preparation, comparison, and training."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from iris_classifier.data import FEATURE_COLUMNS, load_iris_dataframe
from iris_classifier.modeling import (
    build_model_candidates,
    compare_models,
    create_model_comparison_plot,
    select_best_model_name,
    split_dataset,
    train_selected_model,
)


@pytest.fixture(scope="module")
def split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return split_dataset(load_iris_dataframe())


@pytest.fixture(scope="module")
def comparison_results(split_data) -> pd.DataFrame:
    features_train, _, target_train, _ = split_data
    return compare_models(build_model_candidates(), features_train, target_train)


def test_split_dataset_is_reproducible_and_stratified(split_data) -> None:
    features_train, features_test, target_train, target_test = split_data

    assert features_train.shape == (120, 4)
    assert features_test.shape == (30, 4)
    assert tuple(features_train.columns) == FEATURE_COLUMNS
    assert target_train.value_counts().sort_index().tolist() == [40, 40, 40]
    assert target_test.value_counts().sort_index().tolist() == [10, 10, 10]


def test_model_candidates_apply_scaling_where_needed() -> None:
    candidates = build_model_candidates()

    assert set(candidates) == {
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
    }
    assert isinstance(candidates["Logistic Regression"], Pipeline)
    assert isinstance(candidates["K-Nearest Neighbors"], Pipeline)
    assert isinstance(candidates["Decision Tree"], DecisionTreeClassifier)


def test_model_comparison_returns_valid_accuracy_scores(comparison_results) -> None:
    assert len(comparison_results) == 3
    assert comparison_results["mean_validation_accuracy"].between(0, 1).all()
    assert comparison_results["mean_training_accuracy"].between(0, 1).all()
    assert (comparison_results["validation_std"] >= 0).all()


def test_selected_model_can_predict_test_samples(
    split_data,
    comparison_results,
) -> None:
    features_train, features_test, target_train, _ = split_data
    candidates = build_model_candidates()
    selected_name = select_best_model_name(comparison_results)
    estimator = train_selected_model(
        selected_name,
        candidates,
        features_train,
        target_train,
    )

    assert selected_name in candidates
    assert len(estimator.predict(features_test)) == len(features_test)


def test_model_selection_uses_documented_tie_break() -> None:
    tied_results = pd.DataFrame(
        {
            "model": ["Decision Tree", "Logistic Regression"],
            "mean_validation_accuracy": [0.95, 0.95],
        }
    )

    assert select_best_model_name(tied_results) == "Logistic Regression"


def test_model_comparison_plot_is_saved(
    tmp_path: Path,
    comparison_results,
) -> None:
    destination = create_model_comparison_plot(
        comparison_results,
        tmp_path / "model_comparison.png",
    )

    assert destination.exists()
    assert destination.stat().st_size > 0
