# ml_analysis.py
ML analysis for Case Studies in Data Science Individual Task 1 - Decision Tree vs Neural Network classifiers on two healthcare datasets (diabetes, heart failure)
# Healthcare ML Analysis — Individual Task 1, Part 1.3

Supporting code for the Data Analysis section of Individual Task 1 (Case Studies in Data Science, RMIT).

## What this does

Trains and compares two classifiers — a **Decision Tree** and a **Neural Network (MLP)** — on two independent public healthcare datasets, to demonstrate diagnostic/predictive analytics skills relevant to a Data Analyst role.

| Dataset | Source | Task |
|---|---|---|
| Pima Indians Diabetes | [jbrownlee/Datasets](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv) | Predict diabetes diagnosis (768 records) |
| Heart Failure Clinical Records | [UCI ML Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) | Predict mortality during follow-up (299 records) |

## Files

- `ml_analysis.py` — standalone script; trains both models on both datasets, prints metrics and top feature importances
- `healthcare_ml_analysis.ipynb` — same analysis as an executed Jupyter notebook, with outputs retained, for a step-by-step walkthrough

## How to run

```bash
pip install pandas numpy scikit-learn
python ml_analysis.py
```

or open `healthcare_ml_analysis.ipynb` in Jupyter.

## Results summary

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Diabetes | Decision Tree | 0.750 | 0.617 | 0.746 | 0.676 | 0.799 |
| Diabetes | Neural Net | 0.682 | 0.544 | 0.552 | 0.548 | 0.738 |
| Heart Failure | Decision Tree | 0.787 | 0.643 | 0.750 | 0.692 | 0.766 |
| Heart Failure | Neural Net | 0.720 | 0.571 | 0.500 | 0.533 | 0.757 |

Decision Tree outperforms the Neural Network on both datasets across nearly all metrics — likely because both datasets are small (299–768 records), and tree-based models tend to generalise more reliably than neural networks on limited tabular data.

## Issue encountered and fixed

The Neural Network initially collapsed to predicting the majority class (0 precision/recall) on the Diabetes dataset when using the default `adam` solver with early stopping — a known failure mode on small datasets, where the validation split used for early stopping leaves too little data to learn a useful decision boundary. Switching to the `lbfgs` solver (better suited to small datasets per the scikit-learn documentation) with light L2 regularisation (`alpha=0.01`) resolved this and produced stable, non-degenerate results.

## Note on the `time` feature (Heart Failure dataset)

The Decision Tree's top predictor for Heart Failure is `time` (follow-up period length, ~47% importance). This is a known limitation of this dataset: shorter follow-up correlates with the outcome partly due to informative censoring (patients who died were, by definition, not followed for as long), rather than `time` being a genuinely actionable clinical predictor. This is flagged as a data-quality caveat in the report rather than presented as a straightforward insight.
