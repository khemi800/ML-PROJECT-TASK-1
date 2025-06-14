#TASK 1: Credit Scoring Model
#  Objective
#To predict creditworthiness of individuals (i.e., whether they are likely to default or not) using past financial data.

#Step-by-Step Approach

# Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
# Example dataset (replace with your dataset path)
data = pd.read_csv('credit_data.csv')  
print(data.head())

#Preprocessing & Feature Engineering
# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert categorical variables to numeric
data = pd.get_dummies(data, drop_first=True)

# Example: Define features and target
X = data.drop('Creditworthy', axis=1)   # Features (e.g., income, debts, history)
y = data['Creditworthy']                # Target (1 = Good, 0 = Bad)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training & Evaluation
#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))

#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree:\n", classification_report(y_test, y_pred_dt))

#Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# ROC-AUC Score & Curve

y_probs = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test,