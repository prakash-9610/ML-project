import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import joblib

# ==============================
# Load Data
# ==============================
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']
feature_names = X.columns

# ==============================
# Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==============================
# Load scaler
# ==============================
scaler = joblib.load("scaler.joblib")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Load models
# ==============================
lr = joblib.load("logreg_model.joblib")
rf = joblib.load("rf_best_model.joblib")

# ==============================
# Permutation Importance
# ==============================
lr_perm = permutation_importance(
    lr, X_test_scaled, y_test,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc'
)

rf_perm = permutation_importance(
    rf, X_test_scaled, y_test,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc'
)

# Convert to DataFrame
lr_df = pd.DataFrame({
    "feature": feature_names,
    "importance": lr_perm.importances_mean,
    "std": lr_perm.importances_std
}).sort_values(by="importance", ascending=False)

rf_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_perm.importances_mean,
    "std": rf_perm.importances_std
}).sort_values(by="importance", ascending=False)

# ==============================
# Top 15 Features Plot (side by side)
# ==============================
top_n = 15

lr_top = lr_df.head(top_n).iloc[::-1]
rf_top = rf_df.head(top_n).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# LR plot
axes[0].barh(lr_top["feature"], lr_top["importance"], xerr=lr_top["std"])
axes[0].set_title("LR Permutation Importance")
axes[0].set_xlabel("Importance")

# RF plot
axes[1].barh(rf_top["feature"], rf_top["importance"], xerr=rf_top["std"])
axes[1].set_title("RF Permutation Importance")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.show()

# ==============================
# Spearman Rank Correlation
# ==============================
# Rank features
lr_rank = lr_df.set_index("feature").rank(ascending=False)["importance"]
rf_rank = rf_df.set_index("feature").rank(ascending=False)["importance"]

# Align indices
lr_rank = lr_rank.sort_index()
rf_rank = rf_rank.sort_index()

corr, _ = spearmanr(lr_rank, rf_rank)

print("\nSpearman Rank Correlation:", corr)

# ==============================
# Common Features in Top 10
# ==============================
lr_top10 = set(lr_df.head(10)["feature"])
rf_top10 = set(rf_df.head(10)["feature"])

common = lr_top10.intersection(rf_top10)

print("\nCommon features in Top 10:")
for f in common:
    print("-", f)