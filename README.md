# Beta-Bank-Feature-Engineering

Beta Bank Customer Churn Prediction
Feature Engineering • Class Imbalance Handling • Model Optimization • F1 ≥ 0.59

Project Overview

Beta Bank is experiencing customer churn — clients are gradually leaving every month. Since retaining existing customers is significantly cheaper than acquiring new ones, the bank wants a predictive model that identifies customers at risk of leaving.

My Task:
Build a machine learning model that predicts customer churn with an F1 score ≥ 0.59 on the test dataset.  
Additionally, compute and compare the AUC‑ROC metric to evaluate ranking quality.
This project demonstrates:
Feature engineering
Handling missing values
Encoding categorical variables
Managing class imbalance
Model training, tuning, and evaluation
Selecting the best model based on validation performance

Dataset Description

The dataset contains 10,000 customers with the following fields:
Column	Description
RowNumber	Index (not predictive)
CustomerId	Unique customer identifier
Surname	Customer last name
CreditScore	Credit score
Geography	Country (France, Germany, Spain)
Gender	Male/Female
Age	Customer age
Tenure	Years with the bank
Balance	Account balance
NumOfProducts	Number of bank products
HasCrCard	Credit card indicator
IsActiveMember	Activity indicator
EstimatedSalary	Estimated salary
Exited	Target variable (1 = churned, 0 = stayed)


🧹 Data Cleaning & Preparation
1. Missing Values
Tenure had 909 missing values (~9%).
Missingness appeared random → removed rows with missing tenure.
Final dataset size: 9,091 rows.
2. Target Distribution After Cleaning
Stayed (0): 7,237 customers
Churned (1): 1,854 customers
Strong class imbalance → required balancing techniques.
3. Feature Selection
Removed non‑predictive columns (RowNumber, CustomerId, Surname).
Final feature set:
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
4. Encoding
Gender: Label Encoding (Male=1, Female=0)
Geography: One‑Hot Encoding (France, Germany, Spain)
Final feature matrix: 12 columns

Train / Validation / Test Split
You used a three‑way split:
64% training
16% validation
20% test
Stratified sampling preserved churn distribution across splits.

Handling Class Imbalance
Churned customers (class 1) were underrepresented.
You tested three approaches:
✔ 1. Upsampling minority class
Repeated minority class 4× → balanced dataset.
✔ 2. Class weights
RandomForest with class_weight='balanced'.
✔ 3. Hyperparameter tuning
RandomForest tuned with:
n_estimators=300
max_depth=30

Model Training & Comparison
You trained and evaluated four models on the validation set:
Model	Validation F1	Validation AUC
Baseline Random Forest	0.5652	0.8522
RF + Upsampling	0.5909	0.8577
RF + Class Weights	0.5463	0.8611
RF + Upsampling + Tuning	0.6011	0.8596


Winner:
 Random Forest + Upsampling + Hyperparameter Tuning
This model exceeded the required F1 ≥ 0.59 on validation.

Final Test Set Evaluation
After selecting the best model, you evaluated it once on the test set:
Final Metrics
F1 Score: 0.59
AUC‑ROC: 0.8320
Accuracy: 0.85
Classification Report
Class 0 (Stayed): Precision 0.88, Recall 0.94
Class 1 (Churned): Precision 0.68, Recall 0.52
The model performs strongly at ranking customers by churn risk (high AUC‑ROC) and meets the required F1 threshold.

Business Impact
With this model, Beta Bank can:
Identify high‑risk customers early
Launch targeted retention campaigns
Allocate resources efficiently
Reduce churn and increase customer lifetime value
The combination of upsampling + tuned Random Forest provides a reliable, production‑ready churn prediction pipeline

How to Run
Load the dataset (Churn.csv)
Run preprocessing steps
Encode categorical features
Split into train/validation/test
Upsample training data
Train the tuned Random Forest model
Evaluate on validation → then test
