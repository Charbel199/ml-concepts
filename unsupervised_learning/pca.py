from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import time

# Load a sample dataset
data = load_iris()
X = data.data
print(f"Original number of features in dataset: {X.shape[1]}\n")

print("Note: Lower inertia values indicate that data points are closer to their assigned centroids, resulting in more compact clusters.\n")

# **Clustering without PCA**
print("Performing K-Means clustering without PCA...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
start_time = time.time()
kmeans.fit(X)
original_inertia = kmeans.inertia_
original_time = time.time() - start_time
print(f"  - Inertia (without PCA): {original_inertia:.2f}")
print(f"  - Time taken for clustering (without PCA): {original_time:.4f} seconds\n")

# **Dimensionality Reduction** with PCA
print("Reducing dimensionality using PCA...")
start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_time = time.time() - start_time
print(f"  - PCA completed in: {pca_time:.4f} seconds")
print(f"  - Number of features after PCA: {X_pca.shape[1]}\n")

# **Clustering with PCA-reduced data**
print("Performing K-Means clustering on PCA-reduced data...")
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
start_time_pca = time.time()
kmeans_pca.fit(X_pca)
pca_inertia = kmeans_pca.inertia_
pca_clustering_time = time.time() - start_time_pca
print(f"  - Inertia (with PCA): {pca_inertia:.2f}")
print(f"  - Time taken for clustering (with PCA): {pca_clustering_time:.4f} seconds\n")

# Summarizing Results
print("Summary of Results:")
print(f"  - Original clustering inertia: {original_inertia:.2f}")
print(f"  - Clustering inertia after PCA: {pca_inertia:.2f}")
print(f"  - Clustering time without PCA: {original_time:.4f} seconds")
print(f"  - Clustering time with PCA: {pca_clustering_time:.4f} seconds")
print(f"  - Dimensionality reduction time (PCA): {pca_time:.4f} seconds\n")

# **Visualize Clustering Results**
plt.figure(figsize=(10, 5))

# Plot original features (X[:, 0] and X[:, 1])
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("K-Means Clustering (Original Data)")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])

# Plot PCA-reduced features
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pca.labels_, cmap='viridis')
plt.title("K-Means Clustering (PCA-Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()
