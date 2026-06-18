Stroke Prediction Using Machine Learning
The project focuses on predicting stroke risk using healthcare data and Machine Learning techniques.

MSc Data Science Project
Student: Tejandra

Overview
This project uses machine learning algorithms to predict the likelihood of stroke occurrence based on patient demographic and health-related factors. The dataset contains medical and lifestyle information such as age, hypertension, heart disease, BMI, glucose levels, and smoking status.

The objective is to identify individuals at high risk of stroke and compare the effectiveness of different machine learning models.

Dataset
Dataset: Healthcare Stroke Dataset

Features include:

Gender

Age

Hypertension

Heart Disease

Marital Status

Work Type

Residence Type

Average Glucose Level

BMI

Smoking Status

Target Variable:

Stroke (0 = No Stroke, 1 = Stroke)

Research Questions
Can machine learning models accurately predict stroke occurrence using healthcare data?

Which patient characteristics contribute most to stroke prediction?

Does handling class imbalance improve model performance?

Which machine learning algorithm performs best for stroke prediction?

Models Used
Machine Learning Models
XGBoost Classifier

Gaussian Naive Bayes

Decision Tree Classifier

Data Processing Techniques
Label Encoding

StandardScaler

SMOTE (Synthetic Minority Oversampling Technique)

Key Results
Data Preprocessing
Missing BMI values imputed using Median Imputation.

Unknown smoking status values treated as missing and replaced using Mode Imputation.

ID column removed due to lack of predictive value.

Features standardized before model training.

Class Imbalance Handling
The stroke dataset is highly imbalanced.

SMOTE was applied to:

Increase minority class representation.

Improve stroke detection performance.

Reduce model bias toward the majority class.

Model Performance
XGBoost Classifier
Best overall performance

Strong ROC-AUC score

Effective at identifying stroke cases

Robust after hyperparameter tuning

Gaussian Naive Bayes
Fast and simple baseline model

Good computational efficiency

Lower predictive power compared to XGBoost

Decision Tree
Easy to interpret

Provides clear decision rules

Lower generalization performance compared to XGBoost

Best Performing Model
XGBoost achieved the highest overall predictive performance and ROC-AUC score among all tested models.

Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

Confusion Matrix

Visualizations
The project includes:

Stroke Distribution Analysis

Age Distribution

Average Glucose Level Analysis

Correlation Heatmap

Hypertension vs Stroke Comparison

Confusion Matrices

ROC Curve Comparison

How to Run
Install Dependencies
pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn
Run the Notebook
jupyter notebook
Open:

stroke_prediction.ipynb
Project Structure
stroke-prediction/
│
├── healthcare-dataset-stroke-data.csv
├── stroke_prediction.ipynb
├── README.md
│
├── data_preprocessing
├── exploratory_analysis
├── feature_engineering
├── model_training
├── hyperparameter_tuning
└── evaluation
Future Improvements
Implement Random Forest and LightGBM for comparison.

Apply SHAP explainability analysis.

Develop a Streamlit web application.

Deploy the model for real-time stroke risk assessment.

Explore deep learning approaches.

Technologies Used
Python

Pandas

NumPy

Matplotlib

Scikit-learn

XGBoost

Imbalanced-Learn (SMOTE)

License
This project is intended for academic, educational, and research purposes.
