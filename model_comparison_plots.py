import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.metrics import r2_score  # For adding R-squared to the plots

# Set a compatible font for matplotlib and seaborn
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set(font="DejaVu Sans")

# Import the dataset
df = pd.read_csv('Desktop/Vikthore_model_dataset.csv')

# Define universally accepted variable symbols (use plain text for unsupported subscripts)
feature_mnemonics = {
    "Liquid_Viscosity_coefficient (Pa.s)": "μ",
    "Current_Density (A/m^2)": "J",
    "Specific_Resistance (D.m)": "R",
    "Core_Permeability (mO)": "k",
    "Viscosity (Pa.s)": "μ",
    "Pressure_Differential (Pa)": "ΔP",
    "Electric_Potential_Mobile (V)": "ϕ_m",
    "Electric_Potential_Fixed (V)": "ϕ_f",
    "Zeta_Potential (V)": "ζ",
    "Electroosmotic_Velocity (n/s)": "u_eo",
    "Electroosmotic_Flow_Rate (m^3/s)": "q_eo",
    "Darcy_Flow_Rate (m^3/s)": "q_d",
    "Total_Flow_Rate (m^3/s)": "q_t",
    "Porosity (φ)": "φ",
    "Capillary_Pressure (Pc)": "P_c",
    "Relative_Permeability (kr)": "k_r"
}

# Rename the feature columns
df.rename(columns=feature_mnemonics, inplace=True)

# Separate features and target
X = df.iloc[:, :-1]
y = df["Recovery_Factor"]

# Initiate train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train and Predict with all three models ---

# Random Forest Regressor
rfr_model = RFR(n_estimators=100, random_state=42)
rfr_model.fit(X_train, y_train)
y_pred_rfr = rfr_model.predict(X_test)
r2_rfr = r2_score(y_test, y_pred_rfr)

# LightGBM Regressor
lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgbm_model.fit(X_train_scaled, y_train)
y_pred_lgbm = lgbm_model.predict(X_test_scaled)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

# Polynomial SVM
poly_svm_model = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1)
poly_svm_model.fit(X_train_scaled, y_train)
y_pred_svm = poly_svm_model.predict(X_test_scaled)
r2_svm = r2_score(y_test, y_pred_svm)

# --- Create Comparative Plots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better aesthetics

# Plot for Random Forest
axes[0].scatter(y_test, y_pred_rfr, color='skyblue', alpha=0.6, label=f'R-squared: {r2_rfr:.2f}')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
axes[0].set_xlabel("Actual Recovery Factor")
axes[0].set_ylabel("Predicted Recovery Factor")
axes[0].set_title("Random Forest Regressor")
axes[0].legend()
axes[0].grid(True)

# Plot for LightGBM
axes[1].scatter(y_test, y_pred_lgbm, color='lightcoral', alpha=0.6, label=f'R-squared: {r2_lgbm:.2f}')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
axes[1].set_xlabel("Actual Recovery Factor")
axes[1].set_ylabel("Predicted Recovery Factor")
axes[1].set_title("LightGBM Regressor")
axes[1].legend()
axes[1].grid(True)

# Plot for Polynomial SVM
axes[2].scatter(y_test, y_pred_svm, color='lightgreen', alpha=0.6, label=f'R-squared: {r2_svm:.2f}')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
axes[2].set_xlabel("Actual Recovery Factor")
axes[2].set_ylabel("Predicted Recovery Factor")
axes[2].set_title("Polynomial SVM")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

# Save the figure to the desktop (you might need to adjust the path)

plt.savefig("Desktop/Vikkky", format='png', dpi=500)

plt.show()

print(f"Comparative plot saved to: {filepath}")