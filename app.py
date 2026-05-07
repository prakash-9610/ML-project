import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("SHAP Waterfall Plot Demo")

# -----------------------------
# LOAD DATA
# -----------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# SHAP EXPLAINER
# -----------------------------
explainer = shap.Explainer(model, X_train)

# Generate SHAP values
shap_values = explainer(X_test)

# -----------------------------
# SELECT SAMPLE
# -----------------------------
sample_index = st.slider(
    "Select Test Sample",
    0,
    len(X_test) - 1,
    0
)

# -----------------------------
# WATERFALL PLOT
# -----------------------------
st.subheader("SHAP Waterfall Plot")

fig = plt.figure(figsize=(10, 6))

# For binary classification:
# [sample, feature, class]

shap.plots.waterfall(
    shap_values[sample_index, :, 1],
    show=False
)

st.pyplot(fig)

# -----------------------------
# PREDICTION INFO
# -----------------------------
prediction = model.predict(X_test.iloc[[sample_index]])[0]

st.write("Prediction:", prediction)

if prediction == 1:
    st.success("Model predicts: Malignant")
else:
    st.info("Model predicts: Benign")