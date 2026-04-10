import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load dataset
mall_data = pd.read_csv('Mall_Customers.csv')

# Quick look at data
print(mall_data.head())
print(mall_data.info())
print(mall_data.isnull().sum())

# Selecting Annual Income and Spending Score
X = mall_data.iloc[:, [3, 4]].values

# ---------------------------
# Elbow Method
# ---------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
# plt.show()

# ---------------------------
# KMeans Clustering
# ---------------------------
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
kmeans_labels = kmeans.fit_predict(X)

# Silhouette Score
kmeans_score = silhouette_score(X, kmeans_labels)
print("KMeans Silhouette Score:", kmeans_score)

# Plot clusters
plt.figure(figsize=(8, 8))

for i in range(5):
    plt.scatter(X[kmeans_labels == i, 0],
                X[kmeans_labels == i, 1],
                s=50,
                label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=100, c='black', label='Centroids')

plt.title("KMeans Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
# plt.show()

# ---------------------------
# Hierarchical Clustering
# ---------------------------
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
hc_labels = hc.fit_predict(X)

hc_score = silhouette_score(X, hc_labels)
print("Hierarchical Silhouette Score:", hc_score)

# Plot hierarchical clusters
plt.figure(figsize=(8, 8))

for i in range(5):
    plt.scatter(X[hc_labels == i, 0],
                X[hc_labels == i, 1],
                s=50,
                label=f'Cluster {i+1}')

plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
# plt.show()


def predict_customer_cluster(income, score):
    customer = np.array([[income, score]])
    cluster = kmeans.predict(customer)[0]

    print(f"\nCustomer with Income={income} and Score={score}")
    print(f"Assigned to Cluster: {cluster}")

    # Basic interpretation
    if cluster == 0:
        print("Low income - high spending (potential target customers)")
    elif cluster == 1:
        print("High income - high spending (premium customers)")
    elif cluster == 2:
        print("Low income - low spending")
    elif cluster == 3:
        print("High income - low spending")
    elif cluster == 4:
        print("Average customers")

    return cluster


print("\nCluster Centers:")
print(kmeans.cluster_centers_)