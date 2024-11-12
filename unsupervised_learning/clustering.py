from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load a sample dataset
data = load_iris()
X = data.data
print(f"Original data number of features: {X.shape[1]}")

# **Clustering** with K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Create a figure with subplots
plt.figure(figsize=(15, 5))

# Plot for X[:, 0] and X[:, 1]
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering (X[:, 0] vs X[:, 1])")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])

# Plot for X[:, 2] and X[:, 3]
plt.subplot(1, 3, 2)
plt.scatter(X[:, 2], X[:, 3], c=clusters, cmap='viridis')
plt.title("K-Means Clustering (X[:, 2] vs X[:, 3])")
plt.xlabel(data.feature_names[2])
plt.ylabel(data.feature_names[3])

# Plot for X[:, 0] and X[:, 3]
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 3], c=clusters, cmap='viridis')
plt.title("K-Means Clustering (X[:, 0] vs X[:, 3])")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[3])

# Display all subplots
plt.tight_layout()
plt.show()
