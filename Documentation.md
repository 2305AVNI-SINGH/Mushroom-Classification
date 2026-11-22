# Project Documentation: Mushroom Classification

**Author:** Avni Singh
**Dataset:** Mushroom Classification (UCI Machine Learning Repository, via Kaggle)
**Task:** Binary classification, edible vs poisonous

This document explains how the project is organised, how the code works, how to reproduce the results, and the main design decisions. For a short summary and the results, see [README.md](README.md).

> **Disclaimer:** This is an educational project. The model must never be used to decide whether a wild mushroom is safe to eat.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Setup and installation](#3-setup-and-installation)
4. [Data description](#4-data-description)
5. [Pipeline overview](#5-pipeline-overview)
6. [Code reference](#6-code-reference)
7. [Modelling and evaluation](#7-modelling-and-evaluation)
8. [Dashboard](#8-dashboard)
9. [Notebook](#9-notebook)
10. [Assumptions and limitations](#10-assumptions-and-limitations)
11. [Troubleshooting](#11-troubleshooting)
12. [Future improvements](#12-future-improvements)
13. [Glossary](#13-glossary)
14. [Source code](#14-source-code)

---

## 1. Project overview

The goal is to predict whether a mushroom is **edible** or **poisonous** from 22 physical features such as odor, cap shape, gill size and spore print colour. The project covers a full classification workflow:

- Data loading, inspection and cleaning
- Exploratory data analysis (EDA) and visualisation
- One-hot encoding of categorical features inside a pipeline
- Comparison of four models with cross-validation
- Evaluation that focuses on the dangerous mistake (a poisonous mushroom predicted as edible)
- A robustness check on the most informative feature
- An interactive Streamlit dashboard

## 2. Repository structure

```
mushroom-classification/
├── README.md              Short overview, results and findings
├── DOCUMENTATION.md       This document
├── app.py                 Streamlit dashboard
├── requirements.txt       Python dependencies
├── .gitignore             Files excluded from Git
├── data/
│   └── mushrooms.csv      The dataset (small, so it is included)
├── notebooks/
│   └── mushroom_classification.ipynb   EDA, experiments and plots
├── src/
│   ├── preprocess.py      Loading, cleaning and encoding helpers
│   └── train.py           Training and evaluation script
└── outputs/               Created by train.py (results, model, plots)
```

| Path | Purpose |
|---|---|
| `src/preprocess.py` | Data preparation code shared by `train.py` and `app.py` |
| `src/train.py` | Trains and compares the models, saves the best one |
| `app.py` | Loads the saved model and serves the dashboard |
| `notebooks/mushroom_classification.ipynb` | The full exploratory workflow with plots |
| `outputs/` | Generated files. The model file `*.pkl` is excluded from Git |

## 3. Setup and installation

**Requirements:** Python 3.9 or newer.

1. Get the code:
   ```bash
   git clone <your-repo-url>
   cd mushroom-classification
   ```
2. (Optional) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure `data/mushrooms.csv` exists. If not, download it from the [Kaggle dataset page](https://www.kaggle.com/datasets/uciml/mushroom-classification).

**Dependencies:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit.

## 4. Data description

`mushrooms.csv` has **8,124 rows** and **23 columns** (the target `class` plus 22 features). Every column is categorical and holds a single letter. The classes are nearly balanced: 4,208 edible (`e`) and 3,916 poisonous (`p`).

| Group | Features |
|---|---|
| Cap | `cap-shape`, `cap-surface`, `cap-color` |
| General | `bruises`, `odor` |
| Gills | `gill-attachment`, `gill-spacing`, `gill-size`, `gill-color` |
| Stalk | `stalk-shape`, `stalk-root`, `stalk-surface-above-ring`, `stalk-surface-below-ring`, `stalk-color-above-ring`, `stalk-color-below-ring` |
| Veil and ring | `veil-type`, `veil-color`, `ring-number`, `ring-type` |
| Other | `spore-print-color`, `population`, `habitat` |

The meaning of each letter code (for example odor `n` = none, `f` = foul) is stored in the `LABELS` dictionary in `src/preprocess.py` and is used by the dashboard to show readable names.

### Data quality notes

- **No NaN values.** Missing values are written as `?`.
- **`stalk-root`** has about 2,480 `?` values (unknown). They are kept as a separate category called `missing`, because the fact that the value is unknown may itself carry information.
- **`veil-type`** has only one value (`p`) in every row, so it is dropped.

## 5. Pipeline overview

```
mushrooms.csv
      │
 load_data()          read the CSV
      │
   clean()            '?' -> 'missing', drop constant columns
      │
  split_xy()          X = features, y = 1 if poisonous else 0
      │
train_test_split      80% train / 20% test, stratified
      │
 OneHotEncoder        inside a Pipeline, fitted on training data only
      │
 4 classifiers        5-fold cross-validation + test evaluation
      │
 best model           fewest missed poisonous mushrooms
      │
 save                 outputs/ (results, model, plots)
```

## 6. Code reference

### `src/preprocess.py`

| Name | Description |
|---|---|
| `TARGET`, `POISONOUS` | Name of the target column (`class`) and the code for poisonous (`p`) |
| `LABELS` | Dictionary that maps one-letter codes to readable names for every column |
| `load_data(path)` | Reads the CSV into a DataFrame |
| `find_constant_columns(df)` | Lists feature columns that have only one value |
| `clean(df)` | Replaces `?` in `stalk-root` with `missing` and drops constant columns |
| `split_xy(df)` | Returns the features `X` and the target `y` (1 = poisonous, 0 = edible) |
| `make_encoder()` | One-hot encoder that ignores categories it has not seen in training |
| `feature_importance_by_column(pipeline)` | Importance of each original feature, with the one-hot columns summed back together and scaled to add up to 1 |

### `src/train.py`

| Name | Description |
|---|---|
| `get_models(seed)` | The four classifiers, ordered from simplest to most complex |
| `fit_and_score(...)` | Builds an encoder + classifier pipeline, runs cross-validation, fits it, and returns the test metrics |
| `to_markdown(df)` | Formats a results table as Markdown for the README |
| `main(...)` | Runs the whole workflow and saves the outputs |

**Command-line options**

| Option | Default | Meaning |
|---|---|---|
| `--data_path` | `data/mushrooms.csv` | Path to the dataset |
| `--out_dir` | `outputs` | Folder for results, model and plots |
| `--test_size` | `0.2` | Share of data kept for testing |
| `--seed` | `42` | Random seed for reproducible results |

**Files created in `outputs/`:** `results.csv`, `results_extra.csv`, `results.md`, `model.pkl`, `confusion_matrix.png`, `feature_importance.png`.

### `app.py`

A Streamlit app with three tabs:
1. **Predict:** one drop-down per feature (shown with readable names), a Predict button, and the predicted class with the probability of being poisonous.
2. **Data insights:** class balance and the poisonous rate for each value of a chosen feature.
3. **Model results:** the comparison tables from `outputs/`.

## 7. Modelling and evaluation

### Encoding

Every feature is a category, so features are one-hot encoded. The encoder is part of each scikit-learn pipeline, which means it is fitted only on the training data in every cross-validation fold and does not leak information from the test set.

### Validation

- **Split:** 80% train / 20% test, stratified so both sets keep the same edible/poisonous ratio.
- **Cross-validation:** 5-fold stratified on the training set (reported as `CV Accuracy`).
- **Reproducibility:** all random steps use `seed = 42`.

### Models

| Model | Notes |
|---|---|
| Logistic Regression | Simple linear baseline |
| Decision Tree | Easy to interpret |
| Random Forest | 200 trees |
| Gradient Boosting | Boosted shallow trees |

### Metrics

"Poisonous" is the positive class.

| Metric | Meaning |
|---|---|
| **Accuracy** | Share of all mushrooms classified correctly |
| **Precision** | Of mushrooms predicted poisonous, how many really are |
| **Recall** | Of the truly poisonous mushrooms, how many were found. **The most important metric here** |
| **F1** | Balance of precision and recall |
| **ROC AUC** | How well the predicted probabilities rank poisonous above edible |
| **False negatives** | Poisonous mushrooms predicted as edible. The dangerous mistake |
| **False positives** | Edible mushrooms predicted as poisonous. A harmless but wasteful mistake |

### Model selection

Models are ranked by recall, then F1, then CV accuracy. When models tie, the simpler one is chosen because it is easier to explain.

### Robustness check

Odor is a very strong clue. Two extra experiments test the dependence on it:
- **Random Forest without `odor`:** does the model still work when the best clue is removed?
- **Decision Tree with only `odor`:** how far does that one feature go on its own?

## 8. Dashboard

**Start it:**
```bash
python src/train.py --data_path data/mushrooms.csv     # creates outputs/model.pkl
streamlit run app.py
```
Open the address shown in the terminal (usually `http://localhost:8501`).

The dashboard needs `data/mushrooms.csv` and `outputs/model.pkl`. If either is missing, it shows an error asking you to run the training step. It always shows a warning that the model is for education only.

## 9. Notebook

`notebooks/mushroom_classification.ipynb` is self-contained (it does not import from `src/`) and works in Google Colab or Jupyter. Sections:

1. Setup
2. Load the data (in Colab it asks you to upload `mushrooms.csv`)
3. Inspect the data
4. Data cleaning
5. Exploratory data analysis (class balance, poisonous rate by odor, single-feature accuracy)
6. Train/test split
7. Model comparison
8. Best model and confusion matrix
9. Feature importance
10. Robustness check
11. Save the model, results and plots (in Colab, the plots are downloaded as zip files)

The plots are saved to an `images/` folder. Uploading them to the repository is optional.

## 10. Assumptions and limitations

- The dataset describes hypothetical samples of 23 species of gilled mushrooms, as in a field guide. The classes are very easy to separate, so excellent scores here do not show that the problem is easy in the real world.
- New species, or mushrooms with features outside those in the data, may be misclassified.
- The model gives a probability, but it has not been calibrated, so the numbers should not be read as exact risks.
- A single wrong "edible" prediction can be deadly, so this model must not be used for real foraging decisions.
- Results depend on the random split. The cross-validation score gives a sense of the spread.

## 11. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'preprocess'` | Run `python src/train.py` from the project root, and keep `preprocess.py` inside `src/` |
| `FileNotFoundError: ... mushrooms.csv` | Put the file in `data/`, or pass `--data_path` with the correct path |
| Dashboard says "Data or model not found" | Run `python src/train.py --data_path data/mushrooms.csv` first, from the project root |
| `ModuleNotFoundError` for `streamlit` or `sklearn` | Run `pip install -r requirements.txt` |
| Streamlit does not open in Google Colab | Run the dashboard on a local computer, or use the plots from the notebook |

## 12. Future improvements

- Add a model explanation tool such as SHAP values to explain single predictions
- Calibrate predicted probabilities
- Test on a harder, image-based mushroom dataset
- Tune hyperparameters with a grid or random search
- Deploy the dashboard (for example on Streamlit Community Cloud)
- Add automated tests for the preprocessing functions

## 13. Glossary

| Term | Meaning |
|---|---|
| **Categorical feature** | A feature that takes one of a few labels, such as odor = foul |
| **One-hot encoding** | Turning each label into its own 0/1 column so models can use it |
| **Pipeline** | A chain of steps (encoder, then model) fitted together, which prevents data leakage |
| **Stratified split** | A split that keeps the class ratio the same in each part |
| **Cross-validation** | Repeating training and validation on several parts of the data to check the result is stable |
| **Recall** | The share of real positives that the model finds |
| **False negative** | A real positive that the model missed (here, a poisonous mushroom called edible) |
| **Baseline** | A simple model that more complex ones should beat |

---

## 14. Source code

Full source of the project files, exactly as in the repository.

### `src/preprocess.py`

```python
"""Data loading, cleaning and encoding helpers for the Mushroom Classification project."""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

TARGET = "class"      # 'e' = edible, 'p' = poisonous
POISONOUS = "p"

# Meaning of the one-letter codes in the dataset (used by the dashboard)
LABELS = {
    "class": {"e": "edible", "p": "poisonous"},
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green",
                  "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "bruises": {"t": "bruises", "f": "no bruises"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul",
             "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green",
                   "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs",
                   "r": "rooted", "missing": "missing"},
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
                               "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
                               "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
                  "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green",
                          "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered",
                   "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths",
                "u": "urban", "w": "waste", "d": "woods"},
}


def load_data(path):
    """Read mushrooms.csv into a DataFrame."""
    return pd.read_csv(path)


def find_constant_columns(df):
    """Columns with a single value carry no information (e.g. veil-type)."""
    return [c for c in df.columns if c != TARGET and df[c].nunique() == 1]


def clean(df):
    """Clean the raw data.

    - '?' in stalk-root means the value is unknown, so it becomes its own category 'missing'.
    - Columns with only one value are dropped.
    """
    df = df.copy()
    if "stalk-root" in df.columns:
        df["stalk-root"] = df["stalk-root"].replace("?", "missing")
    return df.drop(columns=find_constant_columns(df))


def split_xy(df):
    """Return the features X and the target y (1 = poisonous, 0 = edible)."""
    y = (df[TARGET] == POISONOUS).astype(int)
    X = df.drop(columns=TARGET)
    return X, y


def make_encoder():
    """One-hot encoder for all-categorical features. Unseen categories are ignored."""
    return OneHotEncoder(handle_unknown="ignore")


def feature_importance_by_column(pipeline):
    """Importance of each original feature (one-hot columns are summed back together).

    Works for tree models (feature_importances_) and Logistic Regression (|coefficients|).
    """
    encoder, model = pipeline.steps[0][1], pipeline.steps[-1][1]
    names = encoder.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = abs(model.coef_[0])
    series = pd.Series(values, index=names)
    # One-hot names look like "odor_n"; original column names contain no underscores
    by_column = series.groupby(lambda n: n.split("_")[0]).sum()
    return (by_column / by_column.sum()).sort_values(ascending=False)
```

### `src/train.py`

```python
"""Train and evaluate the mushroom classification models.

Usage:
    python src/train.py --data_path data/mushrooms.csv
"""

import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocess import (clean, feature_importance_by_column, find_constant_columns,
                        load_data, make_encoder, split_xy)


def get_models(seed):
    """Models ordered from simplest to most complex (ties are won by the simpler one)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }


def fit_and_score(name, clf, X_train, X_test, y_train, y_test, cv, with_cv=True):
    """Fit encoder + classifier as one pipeline and compute all test metrics.

    'Poisonous' is the positive class. A false negative means a poisonous mushroom
    was predicted to be edible, which is the dangerous mistake.
    """
    pipe = make_pipeline(make_encoder(), clf)
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean() if with_cv else float("nan")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    row = {
        "Model": name,
        "CV Accuracy": round(cv_acc, 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC AUC": round(roc_auc_score(y_test, proba), 4),
        "False Negatives": int(fn),
        "False Positives": int(fp),
    }
    print(row)
    return row, pipe, pred


def to_markdown(df):
    """Format a table as Markdown so it can be pasted into the README."""
    cols = ["Model"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, row in df.astype(object).iterrows():  # astype(object) keeps integers as integers
        lines.append("| " + " | ".join([name] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def main(data_path, out_dir, test_size, seed):
    df = clean(load_data(data_path))
    print("Dropped constant columns:", find_constant_columns(load_data(data_path)))
    X, y = split_xy(df)

    # Stratified split keeps the edible/poisonous ratio the same in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rows, fitted, predictions = [], {}, {}
    for name, clf in get_models(seed).items():
        row, pipe, pred = fit_and_score(name, clf, X_train, X_test, y_train, y_test, cv)
        rows.append(row)
        fitted[name], predictions[name] = pipe, pred
    results = pd.DataFrame(rows).set_index("Model")

    # Best model: fewest missed poisonous mushrooms (highest recall), then F1, then CV accuracy
    best = results.sort_values(["Recall", "F1", "CV Accuracy"], ascending=False).index[0]
    print("\nBest model:", best)

    # Extra experiments: how much does the model depend on the odor feature?
    extra = []
    row, _, _ = fit_and_score("Random Forest (without odor)", RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1),
        X_train.drop(columns="odor"), X_test.drop(columns="odor"), y_train, y_test, cv)
    extra.append(row)
    row, _, _ = fit_and_score("Decision Tree (odor only)", DecisionTreeClassifier(random_state=seed),
                              X_train[["odor"]], X_test[["odor"]], y_train, y_test, cv)
    extra.append(row)
    extra_results = pd.DataFrame(extra).set_index("Model")

    # Save results, model and plots
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "results.csv"))
    extra_results.to_csv(os.path.join(out_dir, "results_extra.csv"))
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(to_markdown(results) + "\n\nExtra experiments:\n\n" + to_markdown(extra_results) + "\n")
    joblib.dump(fitted[best], os.path.join(out_dir, "model.pkl"))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions[best], display_labels=["Edible", "Poisonous"], cmap="Blues")
    plt.title(f"Confusion matrix: {best}")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    importance = feature_importance_by_column(fitted[best]).head(10)[::-1]
    importance.plot(kind="barh", figsize=(7, 5), title=f"Top 10 features: {best}")
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n", results)
    print("\n", extra_results)
    print(f"\nSaved results, model and plots to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/mushrooms.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.data_path, args.out_dir, args.test_size, args.seed)
```

### `app.py`

```python
"""Streamlit dashboard for the Mushroom Classification project.

Run `python src/train.py --data_path data/mushrooms.csv` first (creates outputs/model.pkl),
then start the dashboard with:
    streamlit run app.py
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocess import LABELS, TARGET, clean, load_data, split_xy  # noqa: E402

DATA_PATH = "data/mushrooms.csv"
MODEL_PATH = "outputs/model.pkl"
RESULTS_PATH = "outputs/results.csv"
EXTRA_PATH = "outputs/results_extra.csv"

st.set_page_config(page_title="Mushroom Classification", layout="wide")
st.title("Mushroom Classification Dashboard")
st.caption("Predict whether a mushroom is edible or poisonous from its physical features.")
st.warning("Educational project only. Never use this model to decide whether a wild "
           "mushroom is safe to eat.")


def name_of(column, code):
    """Readable name for a one-letter code, e.g. odor 'n' -> 'none (n)'."""
    return f"{LABELS.get(column, {}).get(code, code)} ({code})"


try:
    df = clean(load_data(DATA_PATH))
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Data or model not found. Run `python src/train.py --data_path data/mushrooms.csv` first.")
    st.stop()

X, y = split_xy(df)
feature_order = list(model.feature_names_in_)

tab_predict, tab_data, tab_results = st.tabs(["Predict", "Data insights", "Model results"])

# ---- Tab 1: prediction ----
with tab_predict:
    st.subheader("Describe the mushroom")
    st.write("Defaults are the most common value of each feature in the dataset.")
    cols = st.columns(3)
    chosen = {}
    for i, feature in enumerate(feature_order):
        options = sorted(X[feature].unique())
        default = options.index(X[feature].mode()[0])
        chosen[feature] = cols[i % 3].selectbox(
            feature, options, index=default, format_func=lambda c, f=feature: name_of(f, c))

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([chosen])[feature_order]
        p_poison = float(model.predict_proba(row)[0, 1])
        if p_poison >= 0.5:
            st.error(f"Predicted: POISONOUS (probability {p_poison:.1%})")
        else:
            st.success(f"Predicted: edible (probability of poisonous {p_poison:.1%})")
        st.progress(p_poison, text="Probability of poisonous")

# ---- Tab 2: data insights ----
with tab_data:
    st.subheader("Class balance")
    st.bar_chart(df[TARGET].map(LABELS["class"]).value_counts())

    st.subheader("Poisonous rate by feature value")
    feature = st.selectbox("Feature", feature_order, index=feature_order.index("odor")
                           if "odor" in feature_order else 0, key="insight_feature")
    rate = (df.groupby(feature)[TARGET].apply(lambda s: (s == "p").mean() * 100)
            .sort_values(ascending=False))
    rate.index = [name_of(feature, c) for c in rate.index]
    st.bar_chart(rate)
    st.caption("Percentage of mushrooms with this value that are poisonous.")

# ---- Tab 3: model results ----
with tab_results:
    if os.path.exists(RESULTS_PATH):
        st.subheader("Model comparison (test set)")
        st.dataframe(pd.read_csv(RESULTS_PATH, index_col=0))
    if os.path.exists(EXTRA_PATH):
        st.subheader("How much does the model depend on odor?")
        st.dataframe(pd.read_csv(EXTRA_PATH, index_col=0))
    st.caption("False negatives are poisonous mushrooms predicted as edible, the dangerous mistake.")
```

### `requirements.txt`

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

### `.gitignore`

```
# Trained models and generated outputs
*.pkl
outputs/

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/

# OS
.DS_Store
```





  
   
 
 
