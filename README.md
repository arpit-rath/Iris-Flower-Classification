# Iris Flower Classification

A supervised machine learning project that predicts whether an Iris flower is
Setosa, Versicolor, or Virginica from its sepal and petal measurements.

## Project objectives

- Load and validate the Iris dataset.
- Explore relationships between flower measurements.
- Visualize feature distributions and correlations.
- Train and compare classification models.
- Evaluate predictions using accuracy and a confusion matrix.
- Interpret the trained model through suitable visualizations.

## Dataset

The project uses the Iris dataset, which contains 150 observations and four
measurements recorded in centimetres:

- Sepal length
- Sepal width
- Petal length
- Petal width

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)

## Project structure

```text
Iris-Flower-Classification/
|-- notebooks/             # Exploratory analysis and model development
|-- outputs/               # Generated plots and evaluation results
|-- src/iris_classifier/   # Reusable project code
|-- tests/                 # Automated tests
|-- requirements.txt       # Python dependencies
`-- README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Launch JupyterLab and open `notebooks/iris_classification.ipynb`:

```powershell
python -m jupyter lab
```

Run the automated tests:

```powershell
python -m pytest
```

## Development checkpoints

1. Project foundation and dataset validation
2. Exploratory data analysis and visualizations
3. Model training and comparison
4. Evaluation, interpretation, and final documentation

## Modeling approach

The data is divided into an 80% training set and a 20% test set with stratified
sampling. Logistic Regression, Decision Tree, and K-Nearest Neighbors are
compared using five-fold stratified cross-validation on the training data. The
held-out test set is reserved for the final evaluation.
