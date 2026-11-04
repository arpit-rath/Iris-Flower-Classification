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
from iris_classifier.evaluation import (
    EvaluationResult,
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
    create_model_comparison_plot,
    select_best_model_name,
    split_dataset,
    train_selected_model,
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
    "EvaluationResult",
    "create_coefficient_heatmap",
    "create_confusion_matrix_plot",
    "evaluate_model",
    "extract_standardized_coefficients",
    "load_model_bundle",
    "predict_species",
    "save_model_bundle",
    "build_model_candidates",
    "compare_models",
    "create_model_comparison_plot",
    "select_best_model_name",
    "split_dataset",
    "train_selected_model",
]
