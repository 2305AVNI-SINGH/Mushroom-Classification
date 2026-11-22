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
