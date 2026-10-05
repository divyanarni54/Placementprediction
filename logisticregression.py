import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("data.csv")

# 2. Check dataset
print(df.head())
print(df.info())

# 3. Separate input and output
# CHANGE "target" to your actual target column
X = df.drop("target", axis=1)
y = df["target"]

# 4. Keep numerical columns
X = X.select_dtypes(include=["int64", "float64"])

# 5. Handle missing values
X = X.fillna(X.median())

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000
)

# 9. Train model
model.fit(X_train, y_train)

# 10. Predict
pred = model.predict(X_test)

# 11. Accuracy
accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)

# 12. Classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))