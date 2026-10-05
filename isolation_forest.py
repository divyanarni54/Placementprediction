import pandas as pd

from sklearn.ensemble import IsolationForest


df = pd.read_csv("data.csv")

X = df.select_dtypes(include=["number"])

X = X.fillna(X.median())

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

result = model.fit_predict(X)

df["anomaly"] = result

print(df.head())

print("\nAnomalies:")
print(df[df["anomaly"] == -1])