
# Stroke Prediction Using Machine Learning

## Project Overview

This project develops and evaluates machine learning models for predicting stroke occurrence using patient health and demographic data. Early prediction of stroke risk can assist healthcare professionals in identifying high-risk individuals and implementing preventive interventions.

The project follows a complete machine learning pipeline including:

- Data cleaning and preprocessing
- Missing value treatment
- Exploratory Data Analysis (EDA)
- Feature encoding and scaling
- Class imbalance handling using SMOTE
- Model development and evaluation
- Hyperparameter optimization
- Performance comparison using multiple evaluation metrics

---

# Research Objective

The primary objective of this project is to build predictive models capable of identifying stroke risk based on patient characteristics and medical history.

The study aims to:

1. Explore relationships between health indicators and stroke occurrence.
2. Handle class imbalance present in stroke datasets.
3. Compare the performance of multiple machine learning algorithms.
4. Identify the most effective model for stroke prediction.
5. Improve model performance through hyperparameter tuning.

---

# Dataset

### Dataset Name
Healthcare Stroke Dataset

### Target Variable

| Variable | Description |
|-----------|------------|
| stroke | Indicates whether the patient has experienced a stroke (1 = Yes, 0 = No) |

### Features

| Feature | Description |
|-----------|------------|
| gender | Patient gender |
| age | Patient age |
| hypertension | Presence of hypertension |
| heart_disease | Presence of heart disease |
| ever_married | Marital status |
| work_type | Employment type |
| Residence_type | Urban/Rural residence |
| avg_glucose_level | Average glucose level |
| bmi | Body Mass Index |
| smoking_status | Smoking habit information |

---

# Project Workflow

## 1. Data Loading

The dataset is loaded using Pandas and inspected to understand:

- Dataset dimensions
- Data types
- Initial records

---

## 2. Data Cleaning

### Duplicate Handling

- Checked for duplicate records.
- Verified dataset integrity.

### Missing Value Treatment

#### BMI

Missing BMI values were replaced using:

```python
Median Imputation
```

Reason:
- Robust against outliers.
- Suitable for skewed medical data.

#### Smoking Status

Steps performed:

1. Replaced `"Unknown"` values with `NaN`.
2. Filled missing values using the mode.

```python
Most Frequent Value Imputation
```

---

## 3. Feature Selection

The `id` column was removed because:

- It does not contribute to prediction.
- It has no predictive significance.

---

## 4. Exploratory Data Analysis (EDA)

Several visualizations were created to understand data characteristics.

### Target Distribution

- Stroke vs Non-Stroke distribution
- Class imbalance detection

### Age Distribution

- Histogram analysis of patient ages

### Glucose Level Analysis

- Boxplot comparison of glucose levels by stroke occurrence

### Correlation Analysis

- Correlation heatmap for numerical features

### Hypertension Analysis

- Relationship between hypertension and stroke occurrence

---

# Data Preprocessing

## Feature Encoding

Categorical variables were transformed using:

```python
Label Encoding
```

Encoded columns include:

- gender
- ever_married
- work_type
- Residence_type
- smoking_status

---

## Train-Test Split

Dataset was split into:

```text
80% Training Data
20% Testing Data
```

Using:

```python
train_test_split()
```

With:

```python
stratify = y
random_state = 42
```

to preserve class distribution.

---

## Feature Scaling

Features were standardized using:

```python
StandardScaler
```

This ensures:

- Equal feature contribution
- Improved model convergence
- Better model performance

---

## Handling Class Imbalance

Stroke datasets typically contain significantly fewer stroke cases.

To address this issue:

```python
SMOTE (Synthetic Minority Oversampling Technique)
```

was applied only to the training set.

Benefits:

- Balances minority class
- Improves sensitivity to stroke cases
- Reduces prediction bias toward majority class

---

# Machine Learning Models

The following models were implemented and evaluated.

---

## 1. XGBoost Classifier

### Configuration

```python
n_estimators = 200
learning_rate = 0.1
max_depth = 5
random_state = 42
```

### Advantages

- High predictive accuracy
- Handles non-linear relationships
- Robust against overfitting
- Strong performance on tabular healthcare data

---

## 2. Gaussian Naive Bayes

### Advantages

- Fast training
- Computationally efficient
- Effective baseline model
- Performs well with smaller datasets

---

## 3. Decision Tree Classifier

### Configuration

```python
max_depth = 5
random_state = 42
```

### Advantages

- Easy interpretation
- Transparent decision-making process
- Suitable for healthcare applications requiring explainability

---

# Hyperparameter Optimization

To improve predictive performance, Randomized Search Cross-Validation was applied.

---

## XGBoost Tuning

Optimized parameters included:

- Number of estimators
- Learning rate
- Tree depth
- Subsample ratio
- Column sampling ratio
- Gamma
- L1 regularization
- L2 regularization

### Validation Method

```python
Stratified K-Fold Cross Validation
```

with:

```python
n_splits = 5
```

---

## Naive Bayes Tuning

Optimized parameter:

```python
var_smoothing
```

---

## Decision Tree Tuning

Optimized parameters:

- max_depth
- min_samples_split
- min_samples_leaf
- criterion
- max_features

---

# Model Evaluation Metrics

The models were evaluated using:

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-Score

Generated using:

```python
classification_report()
```

---

## Confusion Matrix

Confusion matrices were generated for:

- XGBoost
- Naive Bayes
- Decision Tree

Purpose:

- Evaluate True Positives
- Evaluate False Positives
- Evaluate True Negatives
- Evaluate False Negatives

---

## ROC-AUC Analysis

Receiver Operating Characteristic (ROC) curves were created for:

- XGBoost
- Naive Bayes
- Decision Tree

### Evaluation Metric

```python
ROC-AUC Score
```

A higher AUC value indicates stronger classification performance.

---

# Technologies Used

## Programming Language

- Python

## Libraries

### Data Manipulation

- Pandas
- NumPy

### Visualization

- Matplotlib

### Machine Learning

- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)

### Model Selection

- RandomizedSearchCV
- StratifiedKFold

---

# Project Structure

```text
Stroke-Prediction-Project/
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── notebooks/
│   └── stroke_prediction.ipynb
│
├── outputs/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── evaluation_reports/
│
├── README.md
│
└── requirements.txt
```

---

# Key Contributions

✔ Comprehensive data preprocessing pipeline

✔ Missing value treatment using statistical imputation

✔ Class imbalance handling using SMOTE

✔ Comparative evaluation of three machine learning models

✔ Hyperparameter optimization using Randomized Search

✔ ROC-AUC based performance assessment

✔ Healthcare-focused predictive analytics implementation

---

# Future Improvements

Potential extensions include:

- Random Forest implementation
- LightGBM implementation
- CatBoost implementation
- SHAP explainability analysis
- LIME explainability analysis
- Feature importance visualization
- Deployment using Flask or Streamlit
- Real-time stroke risk prediction system

---

# Conclusion

This project demonstrates a complete machine learning workflow for stroke prediction using healthcare data. By combining robust preprocessing, class balancing techniques, predictive modeling, and hyperparameter optimization, the study provides a framework for identifying individuals at increased risk of stroke.

Among the evaluated models, performance can be further compared using ROC-AUC, precision, recall, and F1-score to determine the most suitable algorithm for healthcare decision-support applications.

---

# Author

**Tejandra**

MSc Data Science

University Dissertation / Machine Learning Project

2026
