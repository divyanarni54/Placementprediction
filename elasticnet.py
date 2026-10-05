import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data.csv")

# Example:
# Change "marks" to your actual target column
X = df.drop("marks", axis=1)
y = df["marks"]

# Keep only numerical columns
X = X.select_dtypes(include=["int64", "float64"])

# Handle missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Elastic Net
model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Predictions:")
print(predictions)

print("Coefficients:")
print(model.coef_)