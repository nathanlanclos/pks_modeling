#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:16:03 2024

@author: jonah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm

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

# ---- Linear Regression ----
print("\n=== Linear Regression ===")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train_continuous)

# Get coefficients and sort by importance
linear_coefficients = linear_model.coef_
linear_coef_df = pd.DataFrame({
    'Feature': combined_features,
    'Coefficient': linear_coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

# Print sorted coefficients
print("\nSorted Coefficients (Linear Regression):")
print(linear_coef_df)

# Evaluate the model
linear_r_squared = linear_model.score(X_test_scaled, y_test_continuous)
print(f"\nR^2 Score on Test Set: {linear_r_squared:.4f}")

# ---- Cook's Distance, Hat Values, and Total Fit ----
# Add a constant to the features for the intercept term in statsmodels
X_train_with_const = sm.add_constant(X_train_scaled)
X_test_with_const = sm.add_constant(X_test_scaled)

# Fit the model using statsmodels to obtain diagnostic metrics
sm_model = sm.OLS(y_train_continuous, X_train_with_const).fit()

# Cook's Distance
cooks_d = sm_model.get_influence().cooks_distance[0]

# Hat Values (Leverage)
hat_values = sm_model.get_influence().hat_matrix_diag

# Total Fit: Residuals vs Fitted Values
fitted_values = sm_model.fittedvalues
residuals = sm_model.resid

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# 1. Cook's Distance
axs[0].stem(np.arange(len(cooks_d)), cooks_d, markerfmt="C0o", use_line_collection=True)
axs[0].set_title("Cook's Distance", fontsize=16, fontweight='bold')
axs[0].set_xlabel("Data Points", fontsize=14, fontweight='bold')
axs[0].set_ylabel("Cook's Distance", fontsize=14, fontweight='bold')

# 2. Hat Values (Leverage)
axs[1].stem(np.arange(len(hat_values)), hat_values, markerfmt="C1o", use_line_collection=True)
axs[1].set_title("Hat Values (Leverage)", fontsize=16, fontweight='bold')
axs[1].set_xlabel("Data Points", fontsize=14, fontweight='bold')
axs[1].set_ylabel("Hat Value", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../Figures/linear_regression_plots.png', dpi=300)
plt.show()