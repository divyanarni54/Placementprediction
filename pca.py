# analysis.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==============================
# 1. Load Dataset
# ==============================

# Change the file name if needed
df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ==============================
# 2. Select Numerical Features
# ==============================

# Select only numerical columns
X = df.select_dtypes(include=["int64", "float64"])

# Remove target column if it exists
target_columns = ["target", "TARGET", "label", "Label"]

for col in target_columns:
    if col in X.columns:
        X = X.drop(columns=[col])


# ==============================
# 3. Handle Missing Values
# ==============================

X = X.fillna(X.median())


# ==============================
# 4. Standardize Data
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# 5. Apply PCA
# ==============================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)


# ==============================
# 6. Explained Variance
# ==============================

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
for i, variance in enumerate(explained_variance):
    print(f"PC{i + 1}: {variance:.4f}")

print(
    "\nTotal variance explained by all components:",
    explained_variance.sum()
)


# ==============================
# 7. Cumulative Variance
# ==============================

cumulative_variance = explained_variance.cumsum()

print("\nCumulative Explained Variance:")
for i, variance in enumerate(cumulative_variance):
    print(f"PC{i + 1}: {variance:.4f}")


# ==============================
# 8. Find Components for 95% Variance
# ==============================

n_components_95 = (
    cumulative_variance >= 0.95
).argmax() + 1

print(
    f"\nNumber of components required for 95% variance: "
    f"{n_components_95}"
)


# ==============================
# 9. Plot Explained Variance
# ==============================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(explained_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% Variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Cumulative Explained Variance")

plt.legend()
plt.grid()

plt.savefig("pca_variance.png")
plt.show()


# ==============================
# 10. PCA with 2 Components
# ==============================

pca_2 = PCA(n_components=2)

X_pca_2 = pca_2.fit_transform(X_scaled)


# ==============================
# 11. Create PCA DataFrame
# ==============================

pca_df = pd.DataFrame(
    X_pca_2,
    columns=["PC1", "PC2"]
)

print("\nPCA Data:")
print(pca_df.head())


# ==============================
# 12. Save PCA Results
# ==============================

pca_df.to_csv(
    "pca_results.csv",
    index=False
)

print("\nPCA results saved to pca_results.csv")


# ==============================
# 13. Plot PC1 vs PC2
# ==============================

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: PC1 vs PC2")

plt.grid()

plt.savefig("pca_2d.png")
plt.show()