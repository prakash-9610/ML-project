# ==============================
# Install required libraries
# ==============================
# Run this once (uncomment if needed)
# !pip install scikit-learn shap lime matplotlib seaborn pandas numpy joblib

# ==============================

# Imports
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import sklearn
import shap
import lime
import lime.lime_tabular

# ==============================
# Set Random Seed
# ==============================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==============================
# Matplotlib Configuration
# ==============================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120

# ==============================
# Print Library Versions
# ==============================
print("Library Versions:")
print("------------------")
print(f"NumPy:        {np.__version__}")
print(f"Pandas:       {pd.__version__}")
print(f"Matplotlib:   {plt.matplotlib.__version__}")
print(f"Seaborn:      {sns.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"SHAP:         {shap.__version__}")
print(f"LIME:         {lime.__version__}")
print(f"Joblib:       {joblib.__version__}")