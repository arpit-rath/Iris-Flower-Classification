"""Utilities for the Iris flower classification project."""

from iris_classifier.data import (
    FEATURE_COLUMNS,
    SPECIES_COLUMN,
    TARGET_COLUMN,
    load_iris_dataframe,
    summarize_dataset,
    validate_iris_dataframe,
)
from iris_classifier.eda import (
    correlation_matrix,
    create_correlation_heatmap,
    create_pairplot,
)

__all__ = [
    "FEATURE_COLUMNS",
    "SPECIES_COLUMN",
    "TARGET_COLUMN",
    "load_iris_dataframe",
    "summarize_dataset",
    "validate_iris_dataframe",
    "correlation_matrix",
    "create_correlation_heatmap",
    "create_pairplot",
]
