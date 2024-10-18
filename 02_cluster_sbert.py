"""
This module performs clustering of product names using SBERT embeddings.
It includes data preprocessing, KMeans clustering, t-SNE visualization,
and saves results for analysis.
"""

# %% Imports
import pandas as pd # pylint: disable=E0401
from sklearn.preprocessing import normalize # pylint: disable=E0401
from sklearn.cluster import KMeans # pylint: disable=E0401
from sklearn.metrics import silhouette_score # pylint: disable=E0401
from sklearn.manifold import TSNE # pylint: disable=E0401
import matplotlib.pyplot as plt # pylint: disable=E0401
from sentence_transformers import SentenceTransformer # pylint: disable=E0401

# Load the dataset
filtered_df = pd.read_csv('filtered_data_with_predictions.csv')
texts = filtered_df['name'].astype(str).tolist()

# Encode product names with SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_tensor=True)

# Normalize embeddings for KMeans
embeddings_normalized = normalize(embeddings.cpu().detach().numpy())

# Define number of clusters (you may adjust based on data analysis)
NUM_CLUSTERS = 10
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
kmeans.fit(embeddings_normalized)

# Add cluster labels to DataFrame
filtered_df['cluster'] = kmeans.labels_

# Add appearances column for frequency of each name within its cluster
filtered_df['appearances'] = filtered_df.groupby(['cluster', 'name'])['name'].transform('count')

# Evaluate Clustering with Silhouette Score
if NUM_CLUSTERS > 1:
    score = silhouette_score(embeddings_normalized, kmeans.labels_)
    print(f"Silhouette Score for {NUM_CLUSTERS} clusters: {score}")

# Visualize clusters with t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(embeddings_normalized)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans.labels_,
                            cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Save clustered data to CSV for further analysis
filtered_df.to_csv('filtered_data_predictions_clusters.csv', index=False)

# %% [markdown]
# Better visual inspection to see how many unique products
# are there in each cluster, and how often they appear.

# %% Display clusters
for cluster_num in range(NUM_CLUSTERS):
    print(f"\nCluster {cluster_num}:")
    cluster_data = filtered_df[filtered_df['cluster'] == cluster_num]

    # Count unique product names within the cluster
    unique_names = cluster_data['name'].value_counts()

    # Print the number of unique names and the names themselves with their counts
    print(f"Number of unique names in Cluster {cluster_num}: {unique_names.shape[0]}")
    print(unique_names.head(10).to_string())  # Display top 10 for brevity; adjust as needed
