"""Dataset loading and validation helpers."""

from typing import Any

import pandas as pd
from sklearn.datasets import load_iris

FEATURE_COLUMNS = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)
TARGET_COLUMN = "target"
SPECIES_COLUMN = "species"
EXPECTED_SPECIES = ("setosa", "versicolor", "virginica")


def load_iris_dataframe() -> pd.DataFrame:
    """Return the Iris dataset as a labelled pandas DataFrame."""
    dataset = load_iris(as_frame=True)
    frame = dataset.frame.copy()
    target_names = dict(enumerate(dataset.target_names))
    frame[SPECIES_COLUMN] = frame[TARGET_COLUMN].map(target_names)
    validate_iris_dataframe(frame)
    return frame


def validate_iris_dataframe(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` when the expected Iris structure is not present."""
    expected_columns = {*FEATURE_COLUMNS, TARGET_COLUMN, SPECIES_COLUMN}
    missing_columns = expected_columns.difference(frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if frame.empty:
        raise ValueError("The Iris dataset is empty.")

    if frame[list(expected_columns)].isnull().any().any():
        raise ValueError("The Iris dataset contains missing values.")

    observed_species = tuple(sorted(frame[SPECIES_COLUMN].unique()))
    if observed_species != tuple(sorted(EXPECTED_SPECIES)):
        raise ValueError(f"Unexpected species labels: {observed_species}")


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return key dataset facts used during the initial inspection."""
    validate_iris_dataframe(frame)
    return {
        "rows": len(frame),
        "features": len(FEATURE_COLUMNS),
        "missing_values": int(frame.isnull().sum().sum()),
        "class_distribution": frame[SPECIES_COLUMN].value_counts().to_dict(),
    }
