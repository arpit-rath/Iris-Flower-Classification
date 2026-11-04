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
|-- models/                # Saved fitted model bundle
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

## Results

All three candidates reached a mean five-fold training-set validation accuracy
of 95.83%. Logistic Regression was selected using the documented simplicity
tie-break and then fitted on all 120 training samples.

On the previously held-out 30-sample test set, the selected model achieved:

- **Accuracy:** 93.33% (28 of 30 predictions correct)
- **Setosa:** 10 of 10 correct
- **Versicolor:** 9 of 10 correct; one predicted as Virginica
- **Virginica:** 9 of 10 correct; one predicted as Versicolor

## Generated artifacts

- `outputs/pairplot.png`
- `outputs/correlation_heatmap.png`
- `outputs/model_comparison.png`
- `outputs/confusion_matrix.png`
- `outputs/coefficient_heatmap.png`
- `models/iris_classifier.joblib`

The saved model bundle contains the fitted preprocessing and classifier
pipeline, feature order, and species labels.

## Limitations

The dataset contains only 150 observations, and the final test set contains 30
flowers. One prediction therefore changes the reported accuracy by 3.33
percentage points. The result demonstrates the project workflow and should not
be treated as evidence of performance on broader real-world flower populations.
