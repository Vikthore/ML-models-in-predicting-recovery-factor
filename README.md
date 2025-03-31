Electrokinetic Enhanced Oil Recovery (EK-EOR) Prediction using Machine Learning

Project Overview
This project explores the application of machine learning models to predict the Recovery Factor (RF) in Electrokinetic Enhanced Oil Recovery (EK-EOR). The dataset was generated synthetically based on key reservoir and electrokinetic parameters, and three different models were trained and evaluated for predictive performance:

LightGBM (LGBM)

Random Forest Regressor (RFR)

Polynomial Support Vector Machine (SVM)

Additionally, a model comparison script is included to visualize and compare the performance of these models.

Project Structure
📁 Project Directory
├── LGBM_Model.py                   # LightGBM model training and evaluation
├── Polynomial_SVM_Model.py          # Polynomial SVM model training and evaluation
├── model_comparison_plots.py        # Generates plots for model performance comparison
├── synthetic_data_generation.py     # Script to generate synthetic dataset
├── README.md                        # Project documentation
1️⃣ Synthetic Data Generation (synthetic_data_generation.py)
This script creates a synthetic dataset incorporating key parameters that influence EK-EOR performance, including:

Zeta potential

Core geometry (length, diameter, area)

Oil viscosity

Electroosmotic permeability

Darcy flow properties

The generated dataset is then used for training and testing machine learning models.

2️⃣ LightGBM Model (LGBM_Model.py)
This script trains and evaluates a LightGBM model, known for its efficiency in handling structured tabular data. It includes:
✔ Data Preprocessing
✔ Model Training
✔ Hyperparameter Tuning
✔ Performance Evaluation (MSE, MAE, R² score)

3️⃣ Polynomial SVM Model (Polynomial_SVM_Model.py)
This script implements a Polynomial Kernel Support Vector Machine (SVM) to predict the Recovery Factor (RF). It follows the same workflow as the LightGBM model but uses an SVM approach.

4️⃣ Model Comparison and Visualization (model_comparison_plots.py)
This script generates performance evaluation plots to compare the models based on:
✔ Mean Squared Error (MSE)
✔ Mean Absolute Error (MAE)
✔ R² Score

Results Summary
Model	Mean Squared Error (MSE)	Mean Absolute Error (MAE)	R² Score
Random Forest Regressor	4.2150e-06	1.7073e-04	0.9999
LightGBM	1.1899e-05	5.5813e-04	0.9983
Polynomial SVM	2.7362e-03	4.5293e-02	0.6101
From the results, Random Forest and LightGBM outperformed the Polynomial SVM model significantly, making them better candidates for predicting Recovery Factor (RF).

Installation & Usage
🔹 Prerequisites
Ensure you have Python 3.8+ installed, along with the required libraries. You can install dependencies using:
pip install -r requirements.txt

🔹 Running the Scripts
1️⃣ Generate Synthetic Data:
python synthetic_data_generation.py

2️⃣ Train and Evaluate LightGBM Model:
python LGBM_Model.py

3️⃣ Train and Evaluate Polynomial SVM Model:
python Polynomial_SVM_Model.py

4️⃣ Compare Model Performances:
python model_comparison_plots.py

Future Work & Improvements
🔹 Incorporate additional EOR parameters for enhanced predictions.
🔹 Implement hyperparameter tuning to optimize all models.
🔹 Explore deep learning approaches (e.g., Neural Networks) for further improvements.
🔹 Integrate the models into a web-based application for real-time predictions.

Contributors
👤 Victor Ikechukwu – Petroleum Engineering & Machine Learning Enthusiast

📧 Contact: vikechukwu@gmail.com

📌 Institution: Federal University of Technology Owerri

🚀 Follow me on GitHub: 
