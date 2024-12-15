#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:16:03 2024

@author: jonah
"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Path to data folder
directory = '../GATSol/dataset/'

# Load CSVs
df_train = pd.read_pickle(directory + 'eSol_train.pkl')
df_test = pd.read_pickle(directory + 'eSol_test.pkl')


continuous_column = 'solubility'  # Replace with your target column name
binary_column = 'binary_solubility'

# Identify scalar features
scalar_features = [
    col for col in df_train.columns
    if pd.api.types.is_numeric_dtype(df_train[col]) and col != continuous_column and col != binary_column
]

# Extract scalar features and target for regression and classification
X_train = df_train[scalar_features]
y_train_continuous = df_train[continuous_column]
y_train_binary = df_train[binary_column]
X_test = df_test[scalar_features]
y_test_continuous = df_test[continuous_column]
y_test_binary = df_test[binary_column]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Linear Regression ----
print("\n=== Linear Regression ===")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train_continuous)

# Get coefficients and sort by importance
linear_coefficients = linear_model.coef_
linear_coef_df = pd.DataFrame({
    'Feature': scalar_features,
    'Coefficient': linear_coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

# Print sorted coefficients
print("\nSorted Coefficients (Linear Regression):")
print(linear_coef_df)

# Evaluate the model
linear_r_squared = linear_model.score(X_test_scaled, y_test_continuous)
print(f"\nR^2 Score on Test Set: {linear_r_squared:.4f}")

# ---- Logistic Regression ----
print("\n=== Logistic Regression ===")
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train_binary)

# Get coefficients and sort by importance
logistic_coefficients = logistic_model.coef_[0]
logistic_coef_df = pd.DataFrame({
    'Feature': scalar_features,
    'Coefficient': logistic_coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

# Print sorted coefficients
print("\nSorted Coefficients (Logistic Regression):")
print(logistic_coef_df)

# Evaluate the model
y_pred_binary = logistic_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nAccuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary))