"""Tests for exploratory analysis calculations and plots."""

from pathlib import Path

import pytest

from iris_classifier.data import FEATURE_COLUMNS, load_iris_dataframe
from iris_classifier.eda import (
    correlation_matrix,
    create_correlation_heatmap,
    create_pairplot,
)


def test_correlation_matrix_has_expected_structure() -> None:
    correlations = correlation_matrix(load_iris_dataframe())

    assert correlations.shape == (4, 4)
    assert tuple(correlations.columns) == FEATURE_COLUMNS
    assert tuple(correlations.index) == FEATURE_COLUMNS
    assert all(correlations.loc[column, column] == pytest.approx(1.0) for column in FEATURE_COLUMNS)


def test_petal_measurements_are_strongly_correlated() -> None:
    correlations = correlation_matrix(load_iris_dataframe())

    assert correlations.loc["petal length (cm)", "petal width (cm)"] > 0.9


@pytest.mark.parametrize(
    ("plot_function", "filename"),
    [
        (create_pairplot, "pairplot.png"),
        (create_correlation_heatmap, "correlation_heatmap.png"),
    ],
)
def test_visualization_is_saved(
    tmp_path: Path,
    plot_function,
    filename: str,
) -> None:
    destination = plot_function(load_iris_dataframe(), tmp_path / filename)

    assert destination.exists()
    assert destination.stat().st_size > 0
