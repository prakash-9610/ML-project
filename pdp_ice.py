# ==============================
# PDP + ICE + 2D Interaction PDP (Clean Version)
# ==============================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ✅ No GUI, prevents crash
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

# ------------------------------
# Load data
# ------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']

feature_names = X.columns.tolist()

# ------------------------------
# Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# Load models
# ------------------------------
scaler = joblib.load("scaler.joblib")
lr_model = joblib.load("logreg_model.joblib")
rf_model = joblib.load("rf_best_model.joblib")

# Scale data
X_train_scaled = scaler.transform(X_train)

# 👉 Use NUMPY to avoid warning
X_train_scaled_np = X_train_scaled

# ------------------------------
# Top 6 features (consensus)
# ------------------------------
top6_features = [
    'worst area',
    'worst concave points',
    'worst radius',
    'worst perimeter',
    'worst concavity',
    'mean concave points'
]

# Convert feature names → indices
top6_idx = [feature_names.index(f) for f in top6_features]

print("Top 6 features:", top6_features)

# ==============================
# PDP + ICE (Logistic Regression)
# ==============================
print("\nGenerating LR PDP + ICE...")

fig, ax = plt.subplots(figsize=(12, 10))

PartialDependenceDisplay.from_estimator(
    lr_model,
    X_train_scaled_np,
    features=top6_idx,
    kind='both',
    subsample=50,
    grid_resolution=50,
    ax=ax
)

plt.suptitle("PDP + ICE (Logistic Regression)")
plt.tight_layout()
plt.savefig("lr_pdp_ice.png")
plt.close()

# ==============================
# PDP + ICE (Random Forest)
# ==============================
print("\nGenerating RF PDP + ICE...")

fig, ax = plt.subplots(figsize=(12, 10))

PartialDependenceDisplay.from_estimator(
    rf_model,
    X_train_scaled_np,
    features=top6_idx,
    kind='both',
    subsample=50,
    grid_resolution=50,
    ax=ax
)

plt.suptitle("PDP + ICE (Random Forest)")
plt.tight_layout()
plt.savefig("rf_pdp_ice.png")
plt.close()

# ==============================
# 2D Interaction PDP (Random Forest)
# ==============================
print("\nGenerating 2D interaction PDP...")

feature_pairs = [
    (feature_names.index('worst area'), feature_names.index('worst radius')),
    (feature_names.index('worst concave points'), feature_names.index('worst concavity')),
    (feature_names.index('worst radius'), feature_names.index('worst perimeter'))
]

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

PartialDependenceDisplay.from_estimator(
    rf_model,
    X_train_scaled_np,
    features=feature_pairs,
    kind='average',
    grid_resolution=30,
    ax=ax
)

plt.suptitle("2D Interaction PDP (Random Forest)")
plt.tight_layout()
plt.savefig("rf_2d_pdp.png")
plt.close()

print("\n✅ All plots saved successfully!")