"""Tests for final evaluation, interpretation, and model persistence."""

from pathlib import Path

import pandas as pd
import pytest

from iris_classifier.data import EXPECTED_SPECIES, load_iris_dataframe
from iris_classifier.evaluation import (
    create_coefficient_heatmap,
    create_confusion_matrix_plot,
    evaluate_model,
    extract_standardized_coefficients,
    load_model_bundle,
    predict_species,
    save_model_bundle,
)
from iris_classifier.modeling import (
    build_model_candidates,
    compare_models,
    select_best_model_name,
    split_dataset,
    train_selected_model,
)


@pytest.fixture(scope="module")
def fitted_model_and_test_data():
    features_train, features_test, target_train, target_test = split_dataset(
        load_iris_dataframe()
    )
    candidates = build_model_candidates()
    results = compare_models(candidates, features_train, target_train)
    selected_name = select_best_model_name(results)
    estimator = train_selected_model(
        selected_name,
        candidates,
        features_train,
        target_train,
    )
    return estimator, features_test, target_test


@pytest.fixture(scope="module")
def evaluation_result(fitted_model_and_test_data):
    estimator, features_test, target_test = fitted_model_and_test_data
    return evaluate_model(estimator, features_test, target_test)


def test_evaluation_covers_all_test_samples(evaluation_result) -> None:
    assert 0 <= evaluation_result.accuracy <= 1
    assert len(evaluation_result.predictions) == 30
    assert evaluation_result.confusion_matrix.shape == (3, 3)
    assert evaluation_result.confusion_matrix.sum() == 30
    assert set(EXPECTED_SPECIES).issubset(evaluation_result.classification_report.index)


def test_coefficients_cover_every_species_and_feature(
    fitted_model_and_test_data,
) -> None:
    estimator, _, _ = fitted_model_and_test_data
    coefficients = extract_standardized_coefficients(estimator)

    assert len(coefficients) == 12
    assert set(coefficients["species"]) == set(EXPECTED_SPECIES)
    assert coefficients["coefficient"].map(pd.notna).all()


@pytest.mark.parametrize(
    ("plot_kind", "filename"),
    [
        ("confusion", "confusion_matrix.png"),
        ("coefficients", "coefficient_heatmap.png"),
    ],
)
def test_evaluation_plot_is_saved(
    tmp_path: Path,
    fitted_model_and_test_data,
    evaluation_result,
    plot_kind: str,
    filename: str,
) -> None:
    estimator, _, _ = fitted_model_and_test_data
    if plot_kind == "confusion":
        destination = create_confusion_matrix_plot(
            evaluation_result,
            tmp_path / filename,
        )
    else:
        coefficients = extract_standardized_coefficients(estimator)
        destination = create_coefficient_heatmap(
            coefficients,
            tmp_path / filename,
        )

    assert destination.exists()
    assert destination.stat().st_size > 0


def test_saved_model_bundle_can_make_prediction(
    tmp_path: Path,
    fitted_model_and_test_data,
) -> None:
    estimator, _, _ = fitted_model_and_test_data
    model_path = save_model_bundle(estimator, tmp_path / "iris_classifier.joblib")
    bundle = load_model_bundle(model_path)

    assert predict_species(bundle, [5.1, 3.5, 1.4, 0.2]) == "setosa"
    assert bundle["feature_names"] == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
