# Mushroom-Classification
# Mushroom Classification: Edible vs Poisonous

## Internship Details

- **Intern ID:** CIS69
- **Full Name:** Avni Singh
- **No. of Weeks:** 1 week
- **Project Name:** Mushroom Classification
- **Project Scope:** Classify mushrooms as edible or poisonous from 22 physical features, covering data cleaning, EDA and visualization, model comparison, evaluation focused on dangerous mistakes (missed poisonous mushrooms), a robustness check, and an interactive prediction dashboard.

## Overview

A binary classification project on the UCI Mushroom dataset. Given a mushroom's physical features (odor, cap, gills, stalk, ring, spore print, habitat and more), the model predicts whether it is **edible** or **poisonous**. Four models are compared with cross-validation, and the evaluation focuses on **recall for the poisonous class**, because predicting "edible" for a poisonous mushroom is the dangerous error.

> **Disclaimer:** This is an educational project. Never use this model to decide whether a wild mushroom is safe to eat.

## Dataset

[Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification) (Kaggle, from the UCI Machine Learning Repository): 8,124 mushrooms described by 22 categorical features, labelled edible (`e`, 4,208) or poisonous (`p`, 3,916). The classes are close to balanced.

Download `mushrooms.csv` from the link above and put it in the `data/` folder. The file is small, so it can be committed to the repository.

## Approach

1. **Cleaning:** `stalk-root` marks unknown values with `?`, which is kept as its own category (`missing`). `veil-type` has a single value and carries no information, so it is dropped.
2. **Exploratory analysis:** class balance, poisonous rate by odor, and how well each single feature separates the classes.
3. **Encoding:** every feature is categorical, so features are one-hot encoded inside a scikit-learn pipeline (no information leaks from the test set).
4. **Split and validation:** stratified 80/20 train/test split, plus 5-fold cross-validation on the training set.
5. **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
6. **Model selection:** the best model misses the fewest poisonous mushrooms (highest recall), then highest F1, then highest CV accuracy. Ties go to the simpler model.
7. **Robustness check:** a Random Forest without the `odor` feature, and a Decision Tree using only `odor`, to see how much the result depends on one feature.

## Results

Test set: 20% of the data (1,625 mushrooms). Positive class = poisonous.

The test set has 842 edible and 783 poisonous mushrooms. The best model (**Random Forest**) is chosen by recall, then F1, then CV accuracy.

| Model | CV Accuracy | Accuracy | Precision | Recall | F1 | ROC AUC | False Negatives | False Positives |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9994 | 0.9994 | 1.0000 | 0.9987 | 0.9994 | 1.0000 | 1 | 0 |
| Decision Tree | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** | **0** |
| Gradient Boosting | 0.9997 | 0.9988 | 1.0000 | 0.9974 | 0.9987 | 1.0000 | 2 | 0 |

**Robustness check**

| Model | Accuracy | Recall | False Negatives |
|---|---|---|---|
| Random Forest (without odor) | 1.0000 | 1.0000 | 0 |
| Decision Tree (odor only) | 0.9858 | 0.9706 | 23 |

## Key findings

- **Class balance:** about 52% edible and 48% poisonous, so accuracy is a fair metric here, but recall for the poisonous class matters most.
- **Model comparison:** all four models reached at least 99.9% cross-validation accuracy. Random Forest and Decision Tree missed no poisonous mushrooms on the test set, Logistic Regression missed 1 and Gradient Boosting missed 2. These differences are only 1 to 2 mushrooms out of 783 poisonous ones, so they are too small to rank the models with confidence.
- **Odor is a strong clue but not safe alone:** a Decision Tree using only `odor` reaches 98.6% accuracy, but it misses 23 poisonous mushrooms (recall 97.1%).
- **Robustness:** a Random Forest trained without `odor` still scores 100% on the test set, so other features carry the same information and the result does not depend on a single feature.
- **Perfect scores come from an easy dataset:** the classes are almost perfectly separable in this data, which does not mean mushroom identification is solved in the real world.

## Limitations

- The dataset is a collection of descriptions of 23 species of gilled mushrooms, not photos or field data, and the classes are very separable, so high scores here do not mean the problem is easy in real life.
- Real mushrooms outside these species could look similar to the ones in the data and be misclassified.
- Even a very small false-negative rate is unsafe when a mistake can be deadly, so the model should never be used for real foraging decisions.

## How to run

```bash
pip install -r requirements.txt
python src/train.py --data_path data/mushrooms.csv
```

This trains the models, prints the results, and saves `outputs/results.csv`, `outputs/results_extra.csv`, `outputs/results.md`, `outputs/model.pkl`, `outputs/confusion_matrix.png` and `outputs/feature_importance.png`.

## Dashboard

An interactive Streamlit app with three tabs: **Predict** (choose a mushroom's features and see the predicted class and probability), **Data insights** (class balance and poisonous rate by any feature), and **Model results**.

```bash
streamlit run app.py
```

Run the training step first, because the dashboard loads the saved model from `outputs/`.

## Project structure

```
mushroom-classification/
├── data/
│   └── mushrooms.csv
├── app.py                 (Streamlit dashboard)
├── notebooks/
│   └── mushroom_classification.ipynb
├── src/
│   ├── preprocess.py      (loading, cleaning, encoding helpers)
│   └── train.py           (training and evaluation)
├── requirements.txt
├── DOCUMENTATION.md
└── README.md
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for the full documentation and source code.

## Tools

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Google Colab
