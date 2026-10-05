import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


df = pd.read_csv("data.csv")

X = df.select_dtypes(include=["number"])
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels = model.fit_predict(X_scaled)

df["cluster"] = labels

print(df.head())

print("\nCluster labels:")
print(labels)