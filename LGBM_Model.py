import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb  # Import LightGBM

# Set a compatible font for matplotlib and seaborn
plt.rcParams["font.family"] = "DejaVu Sans"  # Use a font that supports Unicode symbols
sns.set(font="DejaVu Sans")  # Set the same font for seaborn

# Import the dataset
df = pd.read_csv('Desktop/Vikthore_model_dataset.csv')

# Define universally accepted variable symbols (use plain text for unsupported subscripts)
feature_mnemonics = {
    "Liquid_Viscosity_coefficient (Pa.s)": "μ",      # Viscosity (mu)
    "Current_Density (A/m^2)": "J",                  # Current Density (J)
    "Specific_Resistance (D.m)": "R",                # Resistance (R)
    "Core_Permeability (mO)": "k",                  # Permeability (k)
    "Viscosity (Pa.s)": "μ",                        # Dynamic Viscosity (mu)
    "Pressure_Differential (Pa)": "ΔP",              # Pressure Differential (delta P)
    "Electric_Potential_Mobile (V)": "ϕ_m",          # Mobile Electric Potential (phi_m)
    "Electric_Potential_Fixed (V)": "ϕ_f",           # Fixed Electric Potential (phi_f)
    "Zeta_Potential (V)": "ζ",                      # Zeta Potential (zeta)
    "Electroosmotic_Velocity (n/s)": "u_eo",          # Electroosmotic Velocity (u_eo)
    "Electroosmotic_Flow_Rate (m^3/s)": "q_eo",      # Electroosmotic Flow Rate (q_eo)
    "Darcy_Flow_Rate (m^3/s)": "q_d",                # Darcy Flow Rate (q_d)
    "Total_Flow_Rate (m^3/s)": "q_t",                # Total Flow Rate (q_t)
    "Porosity (φ)": "φ",                            # Porosity (phi)
    "Capillary_Pressure (Pc)": "P_c",                # Capillary Pressure (Pc)
    "Relative_Permeability (kr)": "k_r"              # Relative Permeability (kr)
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

# Initialize the model
lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)  # Initialize LGBM

# Fit the model to the training data
lgbm_model.fit(X_train_scaled, y_train)  # Use scaled training data

# Make predictions
y_pred = lgbm_model.predict(X_test_scaled)  # Use scaled test data

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"The Mean Squared Error (MSE): {mse:.4e}")
print(f"Mean Absolute Error (MAE): {mae:.4e}")
print(f"R_squared score (R2): {r2:.4f}")

# Extract feature importances
feature_importance = lgbm_model.feature_importances_
feature_names = X.columns

# Generate the simplified equation with scientific notation
equation = "Recovery Factor = "
equation_terms = [f"{feature_importance[i]:.3e} * {feature_names[i]}" for i in range(len(feature_names))]
equation += " + ".join(equation_terms)

print("\nSimplified Model Equation:")
print(equation)

# === Enhanced Visualizations === #

# 0. Simple Scatter Plot (Actual vs Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Recovery Factor", fontsize=12)
plt.ylabel("Predicted Recovery Factor", fontsize=12)
plt.title("Actual vs Predicted Recovery Factor(LGBM Model)", fontsize=14)
plt.grid(True)
plt.savefig("Desktop/LGBM1", format = 'png', dpi=800)
plt.show()

# 1. Enhanced Seaborn Regression Plot (Actual vs Predicted)
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
sns.set_palette("husl")

# Scatter plot with regression line
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "dodgerblue", "alpha": 0.6}, line_kws={"color": "crimson", "lw": 2})

# Add a diagonal line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Perfect Prediction")

# Add histograms for actual and predicted values
plt.hist(y_test, bins=30, alpha=0.5, color="blue", label="Actual Recovery Factor")
plt.hist(y_pred, bins=30, alpha=0.5, color="orange", label="Predicted Recovery Factor")

plt.xlabel("Actual Recovery Factor", fontsize=14)
plt.ylabel("Predicted Recovery Factor", fontsize=14)
plt.title("Actual vs Predicted Recovery Factor with Distribution", fontsize=16)
plt.legend(loc="upper left")
plt.savefig("Desktop/LGBM2", format = 'png', dpi=600)
plt.show()

# 2. Enhanced Seaborn Feature Importance Plot
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=True)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create a horizontal bar plot with a gradient color scheme
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis", hue="Feature", legend=False)

# Add annotations for importance values
for index, value in enumerate(feature_importance_df["Importance"]):
    plt.text(value, index, f"{value:.4f}", va="center", fontsize=12, color="black")

plt.xlabel("Feature Importance", fontsize=14)
plt.ylabel("Feature Name", fontsize=14)
plt.title("Feature Importance Plot with Annotations", fontsize=16)
plt.savefig("Desktop/LGBM3", format = 'png', dpi=600)
plt.show()

# 3. Enhanced Plotly Residual Plot
# Ensure y_test and residuals are 1D arrays
residuals = np.array(y_test).flatten() - np.array(y_pred).flatten()

residual_plot = px.scatter(
    x=np.array(y_test).flatten(),  # Ensure 1D array
    y=residuals,                    # Ensure 1D array
    labels={"x": "Actual Recovery Factor", "y": "Residuals"},
    title="Residual Plot with Density Heatmap",
    trendline="lowess",
    trendline_color_override="red",
    opacity=0.6,
    color=abs(residuals),  # Color by absolute residual value
    color_continuous_scale="Viridis"
)

# Add a horizontal zero line
residual_plot.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Residual Line", annotation_position="bottom right")

# Update layout for better visuals
residual_plot.update_layout(
    template="plotly_white",
    xaxis_title="Actual Recovery Factor",
    yaxis_title="Residuals",
    coloraxis_colorbar=dict(title="Absolute Residuals")
)

# Display the plot
residual_plot.show()