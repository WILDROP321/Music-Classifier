import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib



import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_3d_embeddings(embeddings, clusters, method='pca'):
    """
    Plot 3D visualization of song embeddings using PCA or t-SNE.
    
    Args:
        embeddings: np.array of shape (n_samples, n_features)
        clusters: np.array of cluster labels
        method: 'pca' or 'tsne'
    """
    if method == 'pca':
        reducer = PCA(n_components=3)
        reduced = reducer.fit_transform(embeddings)
        title = '3D PCA of Song Embeddings'
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=max(5, min(len(embeddings)//3, 30)), random_state=42)
        reduced = reducer.fit_transform(embeddings)
        title = '3D t-SNE of Song Embeddings'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    fig = go.Figure()

    unique_clusters = np.unique(clusters)

    for cluster_label in unique_clusters:
        idx = clusters == cluster_label
        fig.add_trace(go.Scatter3d(
            x=reduced[idx, 0],
            y=reduced[idx, 1],
            z=reduced[idx, 2],
            mode='markers',
            marker=dict(
                size=6,
                line=dict(width=0.5),
            ),
            name=f'Cluster {cluster_label}'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        ),
        legend_title="Clusters",
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()





# Load embeddings and cluster model
embeddings = np.load('song_embeddings.npy')
kmeans = joblib.load('kmeans_song_cluster.joblib')

# Predict cluster labels
clusters = kmeans.predict(embeddings)

# --- PCA Visualization ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:,0], pca_result[:,1], c=clusters, cmap='tab10', s=10)
plt.title('PCA of Song Embeddings')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.show()

# --- t-SNE Visualization ---
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:,0], tsne_result[:,1], c=clusters, cmap='tab10', s=10)
plt.title('t-SNE of Song Embeddings')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Cluster')
plt.show()


# 3D PCA plot
plot_3d_embeddings(embeddings, clusters, method='pca')

# 3D t-SNE plot (optional, slower but prettier)
plot_3d_embeddings(embeddings, clusters, method='tsne')