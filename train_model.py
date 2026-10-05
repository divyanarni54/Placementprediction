import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. LOAD DATASET
# =========================================================

FILE_NAME = "placement_predict_50k Dataset.csv"

df = pd.read_csv(FILE_NAME)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# =========================================================
# 2. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# =========================================================

columns_to_remove = [
    "StudentID",
    "PlacementStatus",
    "Salary Package",
    "IsAnomaly"
]

X = df.drop(columns=columns_to_remove)

y = df["PlacementStatus"]

# =========================================================
# 3. IDENTIFY COLUMN TYPES
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# =========================================================
# 4. PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numerical_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])

# =========================================================
# 5. MACHINE LEARNING MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# =========================================================
# 6. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])

# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================================================
# 8. TRAIN MODEL
# =========================================================

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")

# =========================================================
# 9. PREDICTION
# =========================================================

y_pred = pipeline.predict(X_test)

# =========================================================
# 10. EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# =========================================================
# 11. SAVE MODEL
# =========================================================

with open(
    "placement_model.pkl",
    "wb"
) as file:

    pickle.dump(
        pipeline,
        file
    )

print("\n===================================")
print("Model saved as placement_model.pkl")
print("===================================")