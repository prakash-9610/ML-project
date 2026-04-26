# ==============================
# LIME Explanations (LR + RF)
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# ------------------------------
# Load data
# ------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']

feature_names = X.columns.tolist()
class_names = ['malignant', 'benign']

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

# ------------------------------
# LIME Explainer
# ------------------------------
explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# ------------------------------
# Predictions
# ------------------------------
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# ------------------------------
# Select samples
# ------------------------------
# High-confidence benign
benign_idx = np.where((lr_prob > 0.9) & (rf_prob > 0.9))[0][:2]

# High-confidence malignant
mal_idx = np.where((lr_prob < 0.1) & (rf_prob < 0.1))[0][:2]

# Uncertain (both models unsure)
# Find most uncertain sample (closest to 0.5)
uncertainty = np.abs(lr_prob - 0.5)
uncertain_idx = np.argmin(uncertainty)

print("Chosen uncertain index:", uncertain_idx)
print("Probability:", lr_prob[uncertain_idx])

print("Selected indices:")
print("Benign:", benign_idx)
print("Malignant:", mal_idx)
print("Uncertain:", uncertain_idx)

# ------------------------------
# Helper: Plot & Save LIME
# ------------------------------
def plot_lime(exp, title, filename):
    vals = exp.as_list()
    features = [x[0] for x in vals]
    weights = [x[1] for x in vals]

    plt.figure(figsize=(6, 4))
    colors = ['green' if w > 0 else 'red' for w in weights]
    plt.barh(features, weights, color=colors)

    plt.title(title)
    plt.xlabel("Contribution")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()

# ------------------------------
# Clean feature names (for comparison)
# ------------------------------
def clean_names(exp):
    cleaned = {}
    for k, v in exp.as_list():
        base = k.split(' ')[0]  # extract feature name
        cleaned[base] = v
    return cleaned

# ------------------------------
# Explain one sample
# ------------------------------
def explain_sample(i):
    print(f"\n--- Sample {i} ---")

    # LR explanation
    exp_lr = explainer.explain_instance(
        X_test_scaled[i],
        lr_model.predict_proba,
        num_features=10
    )
    plot_lime(exp_lr, f"LR Sample {i}", f"lr_sample_{i}.png")

    # RF explanation
    exp_rf = explainer.explain_instance(
        X_test_scaled[i],
        rf_model.predict_proba,
        num_features=10
    )
    plot_lime(exp_rf, f"RF Sample {i}", f"rf_sample_{i}.png")

    return exp_lr, exp_rf

# ------------------------------
# Run LIME
# ------------------------------
for i in benign_idx:
    explain_sample(i)

for i in mal_idx:
    explain_sample(i)

# ------------------------------
# Uncertain sample comparison
# ------------------------------
if uncertain_idx is not None:
    exp_lr, exp_rf = explain_sample(uncertain_idx)

    # Clean feature names
    lr_vals = clean_names(exp_lr)
    rf_vals = clean_names(exp_rf)

    all_features = list(set(lr_vals.keys()).union(set(rf_vals.keys())))

    lr_scores = [lr_vals.get(f, 0) for f in all_features]
    rf_scores = [rf_vals.get(f, 0) for f in all_features]

    y_pos = np.arange(len(all_features))

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos - 0.2, lr_scores, height=0.4, label="Logistic Regression")
    plt.barh(y_pos + 0.2, rf_scores, height=0.4, label="Random Forest")

    plt.yticks(y_pos, all_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Contribution")
    plt.title("LIME Comparison (Uncertain Sample)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("lime_uncertain_comparison.png")
    plt.show()

else:
    print("No uncertain sample found.")