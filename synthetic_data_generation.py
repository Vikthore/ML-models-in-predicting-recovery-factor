import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for triangular distribution (Adjusted)
most_likely_std1, most_likely_std2 = 0.030, 0.035  # Increased most likely for fixed
max_std1, max_std2 = 0.045, 0.045  # Increased max for fixed
min_std1, min_std2 = 0.01, 0.01

# Number of data points
data_points = 100000

# Input Variables Generation (Adjusted current density and liquid viscosity)
liquid_viscosity_coefficient = np.abs(np.random.normal(loc=6, scale=0.45, size=data_points)) # Increased mean
current_density = np.random.normal(loc=0.06, scale=0.01, size=data_points) # Decreased mean
specific_resistance = np.abs(np.random.normal(loc=6.9, scale=0.75, size=data_points))
core_permeability = np.abs(np.random.normal(loc=0.02, scale=0.004, size=data_points))  # Core permeability in mD
viscosity = np.abs(np.random.normal(loc=50, scale=0.5, size=data_points))  # Fluid viscosity in Pa.s
pressure_differential = np.abs(np.random.normal(loc=40, scale=0.65, size=data_points))  # Pressure drop in Pa

# Generate Electric Potentials using Triangular Distribution (Fixed)
electric_potential_mobile = np.random.triangular(min_std1, most_likely_std1, max_std1, size=data_points)
electric_potential_fixed = np.random.triangular(min_std2, most_likely_std2, max_std2, size=data_points)

# Additional Features
porosity = np.random.uniform(0.1, 0.4, size=data_points)  # Porosity (φ)
relative_permeability = np.random.uniform(0.1, 0.9, size=data_points)  # Relative Permeability (kr)

# Capillary Pressure Calculation (Pc)
C = 100  # Scaling constant
capillary_pressure = C * (1 / np.sqrt(core_permeability + 1e-6)) * (1 - porosity) * relative_permeability

# EX-FOR Model Equations
zeta_potential = electric_potential_mobile - electric_potential_fixed

electroosmotic_velocity = (
    (0.1 * current_density * specific_resistance) / (4 * np.pi * liquid_viscosity_coefficient)
) - zeta_potential

electroosmotic_permeability = (0.1 * zeta_potential * viscosity) / (4 * np.pi * liquid_viscosity_coefficient)

electroosmotic_flow_rate = (np.pi * (0.1 / 2) ** 2 * electroosmotic_permeability * current_density * specific_resistance) / viscosity

darcy_flow_rate = (np.pi * (0.1 / 2) ** 2 * core_permeability * pressure_differential) / viscosity


# Total Flow Rate = Electroosmotic Flow + Darcy Flow
total_flow_rate = electroosmotic_flow_rate + darcy_flow_rate

# Recovery Factor Calculation
recovery_factor = darcy_flow_rate - total_flow_rate / darc_flow_rate

# Create the dataset
dataset = pd.DataFrame({
    'Liquid_Viscosity_coefficient (Pa.s)': liquid_viscosity_coefficient,
    'Current_Density (A/m^2)': current_density,
    'Specific_Resistance (D.m)': specific_resistance,
    'Core_Permeability (mO)': core_permeability,
    'Viscosity (Pa.s)': viscosity,
    'Pressure_Differential (Pa)': pressure_differential,
    'Electric_Potential_Mobile (V)': electric_potential_mobile,
    'Electric_Potential_Fixed (V)': electric_potential_fixed,
    'Zeta_Potential (V)': zeta_potential,
    'Electroosmotic_Velocity (n/s)': electroosmotic_velocity,
    'Electroosmotic_Flow_Rate (m^3/s)': electroosmotic_flow_rate,
    'Darcy_Flow_Rate (m^3/s)': darcy_flow_rate,
    'Total_Flow_Rate (m^3/s)': total_flow_rate,
    'Porosity (φ)': porosity,
    'Capillary_Pressure (Pc)': capillary_pressure,
    'Relative_Permeability (kr)': relative_permeability,
    'Recovery_Factor': recovery_factor
})

# Remove rows where Recovery Factor > 0.75
dataset = dataset[dataset['Recovery_Factor'] <= 0.75]

# Convert to a CSV file
dataset.to_csv('Desktop/Vikthore_model_dataset.csv', index=False)

# Print the first few rows of the dataset
print(dataset.head())
