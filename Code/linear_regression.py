#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:03:39 2024

@author: jonah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Path to data folder
directory = '../GATSol/dataset/'

# Load the DataFrame with embeddings
df_train = pd.read_pickle(directory + 'eSol_Train_PCA.pkl')
df_test = pd.read_pickle(directory + 'eSol_Test_PCA.pkl')

# Extract training data
X_train = np.vstack(df_train['pca'].values)
y_train = df_train['solubility']

# Extract test data
X_test = np.vstack(df_test['pca'].values)
y_test = df_test['solubility']

# Initialize and fit the linear regression model on the training data
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on both the training and test sets
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Calculate R^2 scores for both training and test data
r2_train = lr_model.score(X_train, y_train)
r2_test = r2_score(y_test, y_test_pred)

# Visualization for training and test data on the same plot
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', edgecolor='k', alpha=0.6, label='Train Data')
plt.scatter(y_test, y_test_pred, color='green', edgecolor='k', alpha=0.6, label='Test Data')
plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
         [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
         color='red', linestyle='-', linewidth=3, label='Ideal Fit')

# Annotate R^2 scores
plt.text(0.05, 0.95, f'$R^2$ (Train): {r2_train:.3f}', ha='left', va='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='blue')
plt.text(0.05, 0.90, f'$R^2$ (Test): {r2_test:.3f}', ha='left', va='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='green')

# Plot labels and title
plt.xlabel('True Solubility', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Solubility', fontsize=14, fontweight='bold')
plt.title('Linear Regression', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(False)
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig('../Figures/linear_regression.png', dpi=300)