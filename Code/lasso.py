#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:16:03 2024

@author: jonah
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Path to data folder
directory = '../GATSol/dataset/'

# Load DataFrames
df_train = pd.read_pickle(directory + 'eSol_train_updated.pkl')
df_test = pd.read_pickle(directory + 'eSol_test_updated.pkl')

# Remove 'gene' and 'sequence' columns from the DataFrames
df_train.drop(columns=['gene', 'sequence'], inplace=True)
df_test.drop(columns=['gene', 'sequence'], inplace=True)

continuous_column = 'solubility'  # Replace with your target column name
binary_column = 'binary_solubility'

# Identify scalar features (excluding target columns)
scalar_features = [
    col for col in df_train.columns
    if pd.api.types.is_numeric_dtype(df_train[col]) 
    and col != continuous_column 
    and col != binary_column
]

# Function to flatten vector columns into independent features
def flatten_vector_columns(df, vector_columns):
    new_df = df.copy()
    for col in vector_columns:
        # Ensure each entry is a list/array; convert scalars to 1-element lists
        new_df[col] = new_df[col].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else [x])
        
        # Flatten each vector entry and create new columns
        vectors = new_df[col].apply(lambda x: np.array(x).flatten()).tolist()
        max_length = max(len(vec) for vec in vectors)  # Determine the maximum vector length

        # Pad vectors with zeros if needed to match the maximum length
        padded_vectors = [np.pad(vec, (0, max_length - len(vec))) for vec in vectors]

        # Create new columns for each element in the vectors
        new_columns = pd.DataFrame(padded_vectors, columns=[f"{col}_{i}" for i in range(max_length)])
        
        # Concatenate the new columns and drop the original vector column
        new_df = pd.concat([new_df, new_columns], axis=1)
        new_df.drop(columns=[col], inplace=True)
    
    return new_df

# Vector columns to flatten
vector_columns = ['center_of_mass']

# Flatten vector columns in both train and test DataFrames
df_train = flatten_vector_columns(df_train, vector_columns)
df_test = flatten_vector_columns(df_test, vector_columns)

# Combine scalar features and new flattened vector features
combined_features = [col for col in df_train.columns if col not in [continuous_column, binary_column]]

# Extract combined features and targets
X_train = df_train[combined_features].values
y_train_continuous = df_train[continuous_column]
y_train_binary = df_train[binary_column]

X_test = df_test[combined_features].values
y_test_continuous = df_test[continuous_column]
y_test_binary = df_test[binary_column]

# Scale combined features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- LassoCV for Best Alpha ----
print("\n=== LassoCV ===")
# Initialize LassoCV with cross-validation to automatically select the best alpha
lasso_cv_model = LassoCV(cv=5, random_state=42)
lasso_cv_model.fit(X_train_scaled, y_train_continuous)

# Get the best alpha value
best_alpha = lasso_cv_model.alpha_
print(f"Best alpha selected by LassoCV: {best_alpha:.4f}")

# ---- Lasso Regression for Solution Path ----
# Initialize Lasso with multiple alpha values
alphas_lasso = np.logspace(-6, 6, 200)  # Logarithmically spaced alpha values
coefs = []

# Loop through different alphas to calculate the coefficients for each one
for alpha in alphas_lasso:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_scaled, y_train_continuous)
    coefs.append(lasso_model.coef_)

# Convert to numpy array for easy manipulation
coefs = np.array(coefs)

# Plot the solution path
plt.figure(figsize=(10, 6))
plt.plot(alphas_lasso, coefs)
plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'Best alpha: {best_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (log scale)', fontsize=14, fontweight='bold')
plt.ylabel('Coefficients', fontsize=14, fontweight='bold')
plt.title('Lasso Solution Path', fontsize=16, fontweight='bold')

plt.legend()
plt.savefig('../Figures/lasso.png', dpi=300)
plt.show()

# ---- Evaluate LassoCV Model ----
y_pred_continuous = lasso_cv_model.predict(X_test_scaled)

# Calculate R^2 score
lasso_r_squared = lasso_cv_model.score(X_test_scaled, y_test_continuous)
print(f"\nR^2 Score on Test Set (LassoCV): {lasso_r_squared:.4f}")

# Get feature names corresponding to the coefficients
feature_names = df_train[combined_features].columns

# Identify which features have zero coefficients
removed_features = feature_names[lasso_cv_model.coef_ == 0]

# Display removed features
print("Features removed by Lasso regularization:")
print(removed_features)
print(len(removed_features))