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
