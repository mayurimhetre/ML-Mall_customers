# Grouping Mall customers using K-Means Clustering and the Elbow Method

### Dataset
The Mall customers dataset contains information about people visiting the mall. The dataset has gender, customer id, age, annual income, and spending score. It collects insights from the data and group customers based on their behaviors

### Techniques

This repository explains the basics of **K-Means clustering** and how the **Elbow method** helps choose the optimal number of clusters in a dataset.

---

## K-Means Clustering

**K-Means** is an unsupervised machine learning algorithm used to group data points into **K clusters** based on similarity. The goal is to minimize the distance between data points and the cluster centroids.

### How it Works

1. Choose the number of clusters, K.
2. Initialize K centroids randomly.
3. Assign each data point to the nearest centroid.
4. Recalculate the centroids as the mean of assigned points.
5. Repeat steps 3–4 until centroids no longer change significantly.

### Example in Python

```python
from sklearn.cluster import KMeans
import numpy as np

# Example dataset
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Fit K-Means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Cluster labels
print(kmeans.labels_)

# Cluster centroids
print(kmeans.cluster_centers_)
```

Elbow Method

The Elbow method helps determine the optimal number of clusters (K) for K-Means.

How it Works

- Run K-Means for a range of K values (e.g., 1–10).
- Calculate the sum of squared distances (inertia) for each K.
- Plot K vs. inertia.
- The "elbow" point in the graph indicates the optimal K, where adding more clusters does not significantly reduce the inertia.
- Elbow method helps choose the optimal number of clusters by identifying the point where adding more clusters gives diminishing returns.
