import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

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
# Load scaler (from Step 3)
# ==============================
scaler = joblib.load("scaler.joblib")
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==============================
# Baseline Logistic Regression (for ROC overlay)
# ==============================
lr = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

# ==============================
# Random Forest + GridSearchCV
# ==============================
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_
print("Best Params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# ==============================
# Evaluate Best RF
# ==============================
y_pred = best_rf.predict(X_test_scaled)
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

print("\n=== Random Forest Evaluation ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("RF Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# ROC Curves (RF vs LR)
# ==============================
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)

plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc_score(y_test, y_prob):.3f})")
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={roc_auc_score(y_test, lr_prob):.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: RF vs LR")
plt.legend()
plt.show()

# ==============================
# MDI Feature Importances + Error Bars
# ==============================
importances = best_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
    "std": std
}).sort_values(by="importance", ascending=False)

# Plot (all 30 features)
plt.figure(figsize=(10, 8))

plt.barh(
    fi_df["feature"],
    fi_df["importance"],
    xerr=fi_df["std"]
)

plt.xlabel("Mean Decrease in Impurity")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances (MDI) with Error Bars")

plt.gca().invert_yaxis()  # highest importance on top

plt.show()

# ==============================
# Save best RF model
# ==============================
joblib.dump(best_rf, "rf_best_model.joblib")
print("\nBest RF model saved!")