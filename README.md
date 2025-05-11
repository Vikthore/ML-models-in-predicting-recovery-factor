
# Electrokinetic Enhanced Oil Recovery (EK-EOR) Prediction Using Machine Learning

## 🧠 Project Overview

This project applies supervised machine learning algorithms to predict the **Recovery Factor (RF)** in **Electrokinetic Enhanced Oil Recovery (EK-EOR)** processes. Using a **synthetically generated dataset** based on key reservoir and electrokinetic parameters, we evaluate the predictive performance of three models:

- **Random Forest Regressor (RFR)**
- **LightGBM (LGBM)**
- **Polynomial Support Vector Machine (SVM)**

Each model's performance is benchmarked using standard regression metrics and compared through visualized outputs.

## 🎯 Problem Statement

Enhanced Oil Recovery (EOR) through electrokinetics introduces electric fields to improve oil displacement efficiency. However, predicting **Recovery Factor** under such complex multiphysics interactions is challenging. This project explores the feasibility of ML models in capturing these nonlinear dependencies for robust and accurate RF prediction.

## 🛠️ Tools & Technologies

- **Language:** Python 3.8+
- **Libraries:** scikit-learn, LightGBM, matplotlib, pandas, numpy
- **Modeling:** Random Forest, LightGBM, Polynomial SVM
- **Visualization:** matplotlib, seaborn

## 📁 Project Structure

```
EK-EOR-Prediction/
├── LGBM_Model.py                 # LightGBM model training and evaluation
├── Polynomial_SVM_Model.py      # Polynomial SVM model training and evaluation
├── model_comparison_plots.py    # Performance visualization for model comparison
├── synthetic_data_generation.py # Synthetic dataset generation
├── requirements.txt             # Required dependencies
└── README.md                    # Project documentation
```

## 🧪 Methodology

### 1️⃣ Synthetic Data Generation

Features generated include:

- Zeta Potential  
- Oil Viscosity  
- Core Geometry (Length, Diameter, Area)  
- Electroosmotic Permeability  
- Darcy Flow Properties  

This data mimics realistic reservoir conditions and is used for training/testing.

### 2️⃣ Model Training & Evaluation

Each model script includes:

- Data Preprocessing  
- Model Training  
- Metric Evaluation (MSE, MAE, R²)  
- Performance Logging  

### 3️⃣ Model Comparison

`model_comparison_plots.py` plots the comparative performance of all models for a clear visualization of strengths and weaknesses.

## 📊 Results Summary

| Model               | MSE         | MAE         | R² Score |
|--------------------|-------------|-------------|----------|
| Random Forest       | 4.2150e-06  | 1.7073e-04  | 0.9999   |
| LightGBM            | 1.1899e-05  | 5.5813e-04  | 0.9983   |
| Polynomial SVM      | 2.7362e-03  | 4.5293e-02  | 0.6101   |

👉 **Conclusion:** Random Forest and LightGBM exhibit superior accuracy, making them the preferred models for RF prediction in EK-EOR applications.

## ⚙️ Installation & Usage

### 🔹 Prerequisites

Ensure Python 3.8+ and pip are installed.

Install dependencies:
```bash
pip install -r requirements.txt
```

### 🔹 Running the Scripts

```bash
# Step 1: Generate synthetic dataset
python synthetic_data_generation.py

# Step 2: Train LightGBM model
python LGBM_Model.py

# Step 3: Train Polynomial SVM model
python Polynomial_SVM_Model.py

# Step 4: Compare and visualize model performances
python model_comparison_plots.py
```

## 🚀 Future Enhancements

- Integrate more domain-specific parameters (e.g., salinity, pH, wettability)
- Add Bayesian and ensemble stacking methods
- Deploy using Streamlit or Flask for real-time predictions
- Incorporate deep learning models like MLPs or Transformers

## 👤 Contributor

**Victor Ikechukwu**  
Petroleum Engineer & Machine Learning Enthusiast  
📧 Email: vikechukwu@gmail.com  
🎓 Federal University of Technology Owerri  
🔗 [GitHub Profile](https://github.com/Vikthore)
