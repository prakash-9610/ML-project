# ==============================
# Local SHAP: Waterfall + Dependence
# ==============================
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------
# Load data
# ------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']
feature_names = X.columns

# ------------------------------
# Split (same seed)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# Load artifacts
# ------------------------------
scaler = joblib.load("scaler.joblib")
rf_model = joblib.load("rf_best_model.joblib")

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# ------------------------------
# SHAP (TreeExplainer)
# ------------------------------
explainer = shap.TreeExplainer(rf_model)
raw_shap = explainer.shap_values(X_test_scaled)

# Handle all formats
if isinstance(raw_shap, list):
    shap_vals = raw_shap[1]                  # class 1 (benign)
elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
    shap_vals = raw_shap[:, :, 1]
else:
    shap_vals = raw_shap

# Expected value for class 1
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_value = explainer.expected_value[1]
else:
    base_value = explainer.expected_value

# Predictions & probabilities
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]  # prob(benign)

# ------------------------------
# Pick 3 samples
# ------------------------------
# 1) confidently benign (true=1, high prob)
benign_idx = np.where((y_test.values == 1) & (y_prob > 0.9))[0]
idx_benign = benign_idx[0] if len(benign_idx) else np.argmax(y_prob)

# 2) confidently malignant (true=0, low prob)
mal_idx = np.where((y_test.values == 0) & (y_prob < 0.1))[0]
idx_malignant = mal_idx[0] if len(mal_idx) else np.argmin(y_prob)

# 3) misclassified (if exists)
mis_idx = np.where(y_pred != y_test.values)[0]
idx_mis = mis_idx[0] if len(mis_idx) else None

print("Chosen indices:",
      "\n  benign:", idx_benign,
      "\n  malignant:", idx_malignant,
      "\n  misclassified:", idx_mis)

# ------------------------------
# Helper to plot waterfall
# ------------------------------
def plot_waterfall(i, title):
    exp = shap.Explanation(
        values=shap_vals[i],
        base_values=base_value,
        data=X_test_df.iloc[i].values,
        feature_names=feature_names
    )
    shap.waterfall_plot(exp, max_display=15, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Waterfall plots
# ------------------------------
plot_waterfall(idx_benign, "Waterfall: Confidently Benign")

plot_waterfall(idx_malignant, "Waterfall: Confidently Malignant")

if idx_mis is not None:
    plot_waterfall(idx_mis, "Waterfall: Misclassified Sample")
else:
    print("No misclassified sample found (model is very accurate on this split).")

# ------------------------------
# Top 3 features (by mean |SHAP|)
# ------------------------------
mean_abs = np.abs(shap_vals).mean(axis=0)
top3_idx = np.argsort(mean_abs)[-3:][::-1]
top3_features = feature_names[top3_idx]

print("\nTop 3 features (by mean |SHAP|):", list(top3_features))

# ------------------------------
# Dependence plots (with interaction)
# ------------------------------
for feat in top3_features:
    shap.dependence_plot(
        feat,
        shap_vals,
        X_test_df,
        interaction_index="auto",   # color by strongest interaction
        show=False
    )
    plt.title(f"Dependence: {feat}")
    plt.tight_layout()
    plt.show()