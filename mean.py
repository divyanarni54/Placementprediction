from sklearn.impute import SimpleImputer
import numpy as np

# Dataset containing missing values
data = np.array([
    [1, 2, np.nan],
    [4, np.nan, 6],
    [np.nan, 8, 9]
])

# Create an imputer using the mean strategy
imputer = SimpleImputer(strategy='mean')

# Replace missing values with the column mean
imputed_data = imputer.fit_transform(data)

# Display original data
print("Original Data:")
print(data)

# Display data after imputation
print("\nImputed Data:")
print(imputed_data)