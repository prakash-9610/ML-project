import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
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
# Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==============================
# Load scaler from Step 3
# ==============================
scaler = joblib.load("scaler.joblib")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Train Logistic Regression
# ==============================
model = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# ==============================
# Predictions
# ==============================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ==============================
# Metrics
# ==============================
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

# ==============================
# Confusion Matrix Plot
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# ROC Curve
# ==============================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ==============================
# 5-Fold Stratified CV
# ==============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model, scaler.transform(X), y,
    cv=cv, scoring='accuracy'
)

print("\n5-Fold CV Accuracy:", cv_scores)
print("Mean CV Accuracy :", np.mean(cv_scores))

# ==============================
# Coefficient Analysis
# ==============================
coefficients = model.coef_[0]

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
})

# Sort by absolute value
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

# Color: green (positive), red (negative)
colors = ['green' if c > 0 else 'red' for c in coef_df['coefficient']]

# Plot
plt.figure(figsize=(10, 8))
sns.barplot(
    x='coefficient',
    y='feature',
    data=coef_df,
    palette=colors
)

plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

# ==============================
# Save model
# ==============================
joblib.dump(model, "logreg_model.joblib")
print("\nLogistic Regression model saved!")