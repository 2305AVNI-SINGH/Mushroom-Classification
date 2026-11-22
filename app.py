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
