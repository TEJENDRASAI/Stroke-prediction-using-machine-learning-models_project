

# Stroke Prediction Using Machine Learning

The project focuses on predicting stroke risk using Machine Learning and patient health records.

**MSc Data Science Project**  
**Student:** Tejandra Sontineni

---

# Overview

This project uses machine learning techniques to predict whether a patient is at risk of experiencing a stroke based on demographic and health-related factors.

The workflow includes:

- Data cleaning and preprocessing
- Missing value handling
- Exploratory Data Analysis (EDA)
- Feature encoding and scaling
- Class imbalance handling using SMOTE
- Machine learning model training
- Hyperparameter tuning
- Model evaluation and comparison

---

# Dataset

**Dataset:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Features

- Gender
- Age
- Hypertension
- Heart Disease
- Marital Status
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

## Target Variable

- **Stroke**
  - 1 = Stroke
  - 0 = No Stroke

---

# Research Objectives

- Identify factors associated with stroke occurrence.
- Compare multiple machine learning models for stroke prediction.
- Improve predictive performance through hyperparameter tuning.
- Evaluate models using healthcare-related performance metrics.

---

# Models Used

## XGBoost Classifier

- High predictive performance
- Handles non-linear relationships effectively
- Robust against overfitting

## Gaussian Naive Bayes

- Fast and computationally efficient
- Useful baseline model

## Decision Tree Classifier

- Easy to interpret
- Suitable for explainable healthcare applications

---

# Data Preprocessing

## Missing Value Treatment

### BMI

- Median Imputation

### Smoking Status

- Mode Imputation

## Feature Encoding

- Label Encoding

## Feature Scaling

- StandardScaler

## Class Imbalance Handling

- SMOTE (Synthetic Minority Oversampling Technique)

---

# Hyperparameter Tuning

The following techniques were used:

- RandomizedSearchCV
- Stratified K-Fold Cross Validation (5-Fold)

---

# Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

# Key Findings

- SMOTE improved minority class prediction.
- Hyperparameter tuning improved overall model performance.
- XGBoost achieved the strongest predictive performance among the evaluated models.
- Age, glucose level, BMI, hypertension, and heart disease were important indicators of stroke risk.

---

# Technologies Used

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)

---

# Project Structure

```text
Stroke-Prediction/
│
├── stroke_prediction.ipynb
├── healthcare-dataset-stroke-data.csv
├── README.md
└── requirements.txt
```

---

# How to Run

## Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn
```

## Launch Jupyter Notebook

```bash
jupyter notebook
```

## Open the Notebook

```text
stroke_prediction.ipynb
```

---

# Future Improvements

- Random Forest implementation
- LightGBM implementation
- SHAP explainability analysis
- LIME explainability analysis
- Streamlit deployment
- Real-time stroke risk prediction system

---
