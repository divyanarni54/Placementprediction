import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("data.csv")

X = df.select_dtypes(include=["number"])

X = X.fillna(X.median())

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
)

labels = model.fit_predict(X_scaled)

df["cluster"] = labels

print(df.head())

print("\nCluster Centers:")
print(model.cluster_centers_)