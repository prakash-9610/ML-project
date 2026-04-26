# ==============================
# Consensus Feature Ranking
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

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

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==============================
# 1. LR Coefficients
# ==============================
lr_coef = np.abs(lr_model.coef_[0])

# ==============================
# 2. RF MDI
# ==============================
rf_mdi = rf_model.feature_importances_

# ==============================
# 3. LR Permutation Importance
# ==============================
lr_perm = permutation_importance(
    lr_model,
    X_test_scaled,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc'
)

lr_perm_vals = lr_perm.importances_mean

# ==============================
# 4. RF Permutation Importance
# ==============================
rf_perm = permutation_importance(
    rf_model,
    X_test_scaled,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc'
)

rf_perm_vals = rf_perm.importances_mean

# ==============================
# Create DataFrame
# ==============================
importance_df = pd.DataFrame({
    'feature': feature_names,
    'LR_coef': lr_coef,
    'RF_MDI': rf_mdi,
    'LR_perm': lr_perm_vals,
    'RF_perm': rf_perm_vals
})

# ==============================
# Convert to RANKS (1 = best)
# ==============================
rank_df = importance_df.copy()

for col in ['LR_coef', 'RF_MDI', 'LR_perm', 'RF_perm']:
    rank_df[col] = rank_df[col].rank(ascending=False)

# ==============================
# Mean rank & std
# ==============================
rank_df['mean_rank'] = rank_df[['LR_coef','RF_MDI','LR_perm','RF_perm']].mean(axis=1)
rank_df['std_rank'] = rank_df[['LR_coef','RF_MDI','LR_perm','RF_perm']].std(axis=1)

# Sort by consensus
rank_df = rank_df.sort_values(by='mean_rank')

# ==============================
# Top 10 consensus features
# ==============================
print("\nTop 10 Consensus Features:\n")

top10 = rank_df.head(10)

for i, row in top10.iterrows():
    print(f"{row['feature']}: mean_rank={row['mean_rank']:.2f}, std={row['std_rank']:.2f}")

# ==============================
# Heatmap (Top 15 × 4 methods)
# ==============================
top15 = rank_df.head(15)

heatmap_data = top15.set_index('feature')[['LR_coef','RF_MDI','LR_perm','RF_perm']]

plt.figure(figsize=(10, 8))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap='coolwarm_r',
    fmt=".1f"
)

plt.title("Feature Ranking Heatmap (Top 15)")
plt.xlabel("Methods")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("consensus_heatmap.png")
plt.close()

print("\n✅ Heatmap saved as 'consensus_heatmap.png'")