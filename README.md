# Fraud-Detection-Program
A machine learning pipeline to detect fraudulent financial transactions using a real-world dataset with over 6.3 million records. Includes EDA, feature engineering, and model training.

# Overview

This project presents a robust and scalable machine learning pipeline designed to detect fraudulent financial transactions with high accuracy and reliability. Using a real-world dataset of over 6.3 million records from Kaggle, the pipeline walks through the full data science workflow — from data ingestion and exploration to feature engineering, model training, and evaluation.

The model is trained to identify suspicious behavior in transaction data such as transfers and cash outs, leveraging balance movements and transaction metadata. After a series of rigorous evaluations, the final model achieves 94% accuracy, demonstrating strong performance in identifying fraudulent activity in imbalanced datasets.

The objective is to predict whether a given transaction is fraudulent based on features like transaction type, amount, balances, and more. This project covers the full data science lifecycle: from data wrangling to model deployment-ready output.

# 📂 Dataset

Source: Kaggle (https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)

Size: 6,362,620 records

Key Features:

type: Type of transaction (e.g., TRANSFER, CASH_OUT)

amount: Transaction amount

oldbalanceOrg & newbalanceOrig: Sender's balances before and after transaction

oldbalanceDest & newbalanceDest: Receiver's balances before and after transaction

isFraud: Label (1 for fraud, 0 for normal)

# 🔁 Project Workflow

Data Import & Cleaning

Loaded dataset using Pandas

Removed irrelevant columns

Handled missing and zero-balance edge cases

Exploratory Data Analysis (EDA)

Analyzed distribution of transaction types and fraud occurrence

Identified correlation among features

Visualized balance changes and suspicious patterns

Feature Engineering

Created new features like balance_diff, is_same_sender_receiver, etc.

Normalized amount-based features

Modeling & Evaluation

Trained classification models (Random Forest, Logistic Regression)

Evaluated on metrics like Precision, Recall, F1-score, ROC-AUC

Saved the best model as fraud_detection_pipeline.pkl

# 📊 Exploratory Data Analysis

Fraudulent transactions are concentrated in TRANSFER and CASH_OUT types

Imbalance between fraud and non-fraud classes (required special care during modeling)

High-value transactions are disproportionately likely to be fraudulent

# 🧠 Feature Engineering & Selection

Removed nameOrig, nameDest due to lack of predictive power

Engineered:

errorBalanceOrig = oldbalanceOrg - newbalanceOrig - amount

errorBalanceDest = newbalanceDest - oldbalanceDest - amount

Used SelectKBest and correlation filtering for final feature set

# 🤖 Modeling Approach

Algorithms tried:

Logistic Regression

Random Forest Classifier (final pick)

Why Random Forest:

Better handles class imbalance

High recall without overfitting

Handling Imbalance:

Used stratified sampling and adjusted class weights

# 📈 Evaluation Metrics

Accuracy: Overall correctness

Precision: Fraction of predicted frauds that were actual frauds

Recall: Fraction of real frauds that were detected

F1 Score: Balance between Precision and Recall

ROC-AUC: Performance across thresholds

🛠️ How to Run the Project

# Requirements:

Python 3.7+

Install dependencies:
#pip install pandas numpy scikit-learn matplotlib seaborn

Run the Notebook
#python fraud_detection.py

Load the Trained Model
#import joblib
model = joblib.load("fraud_detection_pipeline.pkl")


# ✅ Results

Achieved high precision and recall on fraudulent transactions

Demonstrated reliability even with unbalanced dataset

Model exported for future deployment

# 🔮 Future Improvements

Apply SVD or PCA to reduce feature space
<img width="1512" alt="Screenshot 2025-06-01 at 11 57 33 PM" src="https://github.com/user-attachments/assets/083b6482-197e-41ca-bd3a-11e75b994ffe" />

Use advanced models like XGBoost or LightGBM

Deploy with Streamlit or FastAPI for interactive use

Incorporate real-time streaming with Kafka or Spark

# 🧰 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook


# ✍️ Author

Aryan Raj (B.S. in Data Science) 
LinkedIn: www.linkedin.com/in/aryan-raj-a742bb203

<img width="1507" alt="Screenshot 2025-06-01 at 11 59 34 PM" src="https://github.com/user-attachments/assets/0439052d-89c8-46f8-a95c-571b56839659" />


