"""Tests for Iris dataset loading and validation."""

import pandas as pd
import pytest

from iris_classifier.data import (
    FEATURE_COLUMNS,
    SPECIES_COLUMN,
    TARGET_COLUMN,
    load_iris_dataframe,
    summarize_dataset,
    validate_iris_dataframe,
)


def test_load_iris_dataframe_has_expected_shape() -> None:
    frame = load_iris_dataframe()

    assert frame.shape == (150, 6)
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert TARGET_COLUMN in frame.columns
    assert SPECIES_COLUMN in frame.columns


def test_dataset_is_balanced_and_complete() -> None:
    summary = summarize_dataset(load_iris_dataframe())

    assert summary["rows"] == 150
    assert summary["features"] == 4
    assert summary["missing_values"] == 0
    assert summary["class_distribution"] == {
        "setosa": 50,
        "versicolor": 50,
        "virginica": 50,
    }


def test_validation_rejects_missing_required_column() -> None:
    frame = load_iris_dataframe().drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_iris_dataframe(frame)


def test_validation_rejects_missing_values() -> None:
    frame = load_iris_dataframe()
    frame.loc[0, FEATURE_COLUMNS[0]] = pd.NA

    with pytest.raises(ValueError, match="contains missing values"):
        validate_iris_dataframe(frame)
