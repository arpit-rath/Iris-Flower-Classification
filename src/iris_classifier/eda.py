"""Exploratory data analysis and visualization helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from iris_classifier.data import (
    FEATURE_COLUMNS,
    SPECIES_COLUMN,
    validate_iris_dataframe,
)

SPECIES_PALETTE = {
    "setosa": "#2A9D8F",
    "versicolor": "#E9C46A",
    "virginica": "#E76F51",
}


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlations between the four flower measurements."""
    validate_iris_dataframe(frame)
    return frame.loc[:, list(FEATURE_COLUMNS)].corr()


def create_pairplot(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Create and save a species-coloured pairplot."""
    validate_iris_dataframe(frame)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.pairplot(
        frame,
        vars=list(FEATURE_COLUMNS),
        hue=SPECIES_COLUMN,
        hue_order=list(SPECIES_PALETTE),
        palette=SPECIES_PALETTE,
        diag_kind="hist",
        plot_kws={"alpha": 0.75, "s": 45, "edgecolor": "white"},
        diag_kws={"alpha": 0.65},
    )
    grid.figure.suptitle("Iris Feature Relationships by Species", y=1.02)
    grid.figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(grid.figure)
    return destination


def create_correlation_heatmap(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create and save an annotated feature-correlation heatmap."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    correlations = correlation_matrix(frame)

    sns.set_theme(style="white", context="notebook")
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlations,
        annot=True,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    axis.set_title("Correlation Between Iris Measurements", pad=14)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination
