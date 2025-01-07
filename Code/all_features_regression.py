import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm  # For OLS to calculate p-values

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

# --- Calculate p-values using statsmodels ---
X_train_scaled_with_const = sm.add_constant(X_train_scaled)  # Add constant term for intercept

# Fit the model
ols_model = sm.OLS(y_train_continuous, X_train_scaled_with_const).fit()

# Get the p-values for each feature
p_values = ols_model.pvalues[1:]  # Exclude intercept

# Create a DataFrame to display coefficients with p-values
linear_coef_df = pd.DataFrame({
    'Feature': combined_features,
    'Coefficient': linear_coefficients
})

# Sort coefficients by absolute value for easy interpretation

# Add p-values to the DataFrame (ensuring correct order)
linear_coef_df['P-value'] = p_values.values  # Ensure values are assigned correctly
linear_coef_df = linear_coef_df.reindex(linear_coef_df['P-value'].abs().sort_values(ascending=False).index)

# Print coefficients with p-values
print("\nSorted Coefficients with p-values (Linear Regression):")
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
    'Feature': combined_features,
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

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test_binary, y_pred_binary)

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: Insoluble', 'Pred: Soluble'], yticklabels=['True: Insoluble', 'True: Soluble'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('../Figures/confusion_matrix.png', dpi=300)
plt.show()