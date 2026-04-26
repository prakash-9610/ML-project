# ==============================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# Load Dataset
# ==============================
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Map target to labels
df['target_label'] = df['target'].map({0: 'malignant', 1: 'benign'})

# ==============================
# Basic Info
# ==============================
print("Shape of dataset:", df.shape)

print("\nClass Distribution:")
print(df['target_label'].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# ==============================
# 1. Class Distribution Plot
# ==============================
plt.figure()
sns.countplot(x='target_label', data=df)
plt.title("Class Distribution (Malignant vs Benign)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# ==============================
# 2. Correlation Heatmap
# ==============================
plt.figure(figsize=(12, 10))
corr = df.drop(columns=['target', 'target_label']).corr()

sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# ==============================
# 3. KDE Plots for Top 6 Features
# ==============================

# Get top 6 features correlated with target
corr_target = df.corr(numeric_only=True)['target'].abs().sort_values(ascending=False)
top_features = corr_target.index[1:7]  # skip target itself

print("\nTop 6 features:", list(top_features))

for feature in top_features:
    plt.figure()
    sns.kdeplot(data=df, x=feature, hue='target_label', fill=True)
    plt.title(f"KDE Plot of {feature} by Class")
    plt.show()

# ==============================
# Split into X and y
# ==============================
X = df.drop(columns=['target', 'target_label'])
y = df['target']

feature_names = X.columns.tolist()

# ==============================
# Train/Test Split (80/20, stratified)
# ==============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# Standard Scaling (fit ONLY on train)
# ==============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train
X_test_scaled = scaler.transform(X_test)         # transform test

# ==============================
# Verify class balance
# ==============================
def show_distribution(name, y_data):
    dist = y_data.value_counts(normalize=True).sort_index()
    print(f"{name} distribution:")
    for cls, ratio in dist.items():
        print(f"  Class {cls}: {ratio:.4f}")
    print()

show_distribution("Original", y)
show_distribution("Train", y_train)
show_distribution("Test", y_test)

# ==============================
# Save scaler and feature names
# ==============================
joblib.dump(scaler, "scaler.joblib")
joblib.dump(feature_names, "feature_names.joblib")

print("Scaler and feature names saved successfully!")