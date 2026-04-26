# ==============================
# SHAP for LR & RF (Robust Version)
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
# Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# Load artifacts
# ------------------------------
scaler = joblib.load("scaler.joblib")
lr_model = joblib.load("logreg_model.joblib")
rf_model = joblib.load("rf_best_model.joblib")

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert to DataFrame (important for SHAP)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# ==============================
# 1) Logistic Regression SHAP
# ==============================
print("\n--- Logistic Regression SHAP ---")

lr_explainer = shap.LinearExplainer(
    lr_model,
    X_train_scaled,
    feature_perturbation="interventional"
)

lr_shap_values = lr_explainer.shap_values(X_test_scaled)

print("LR SHAP shape:", np.array(lr_shap_values).shape)

# --- Beeswarm
shap.summary_plot(lr_shap_values, X_test_df)
plt.show()

# --- Bar Plot
shap.summary_plot(lr_shap_values, X_test_df, plot_type="bar")
plt.show()

# ==============================
# 2) Random Forest SHAP
# ==============================
print("\n--- Random Forest SHAP ---")

rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_test_scaled)

# ✅ Handle ALL SHAP formats safely
if isinstance(rf_shap_values, list):
    rf_shap_values_class1 = rf_shap_values[1]
elif isinstance(rf_shap_values, np.ndarray) and len(rf_shap_values.shape) == 3:
    rf_shap_values_class1 = rf_shap_values[:, :, 1]
else:
    rf_shap_values_class1 = rf_shap_values

print("RF SHAP shape:", rf_shap_values_class1.shape)
print("X_test shape:", X_test_df.shape)

# --- Beeswarm
shap.summary_plot(rf_shap_values_class1, X_test_df)
plt.show()

# --- Bar Plot
shap.summary_plot(rf_shap_values_class1, X_test_df, plot_type="bar")
plt.show()

# ==============================
# 3) Top Feature Comparison
# ==============================
def get_top_features(shap_vals, feature_names):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values(by="mean_abs_shap", ascending=False)
    return df_imp

lr_imp = get_top_features(lr_shap_values, feature_names)
rf_imp = get_top_features(rf_shap_values_class1, feature_names)

lr_top10 = lr_imp.head(10)
rf_top10 = rf_imp.head(10)

print("\nTop 10 (LR):")
print(lr_top10["feature"].tolist())

print("\nTop 10 (RF):")
print(rf_top10["feature"].tolist())

# ------------------------------
# Common features
# ------------------------------
common = set(lr_top10["feature"]).intersection(set(rf_top10["feature"]))

print("\nCommon features in Top 10:")
for f in common:
    print("-", f)

# ==============================
# 4) Side-by-side comparison plot
# ==============================
union_feats = list(set(lr_top10["feature"]).union(set(rf_top10["feature"])))

plot_df = pd.DataFrame({
    "feature": union_feats,
    "LR": [lr_imp.set_index("feature").loc[f, "mean_abs_shap"] for f in union_feats],
    "RF": [rf_imp.set_index("feature").loc[f, "mean_abs_shap"] for f in union_feats],
}).sort_values(by="LR", ascending=False)

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(plot_df))

plt.barh(y_pos - 0.2, plot_df["LR"], height=0.4, label="Logistic Regression")
plt.barh(y_pos + 0.2, plot_df["RF"], height=0.4, label="Random Forest")

plt.yticks(y_pos, plot_df["feature"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP|")
plt.title("SHAP Feature Importance Comparison (LR vs RF)")
plt.legend()

plt.tight_layout()
plt.show()