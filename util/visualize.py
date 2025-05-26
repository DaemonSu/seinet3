import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns


def visualize_features(features, labels, known_class_count, method='tsne', title=''):
    """
    Visualize feature embeddings using t-SNE or UMAP.

    Args:
        features (np.ndarray): Shape (N, D) feature matrix.
        labels (np.ndarray): Shape (N,) integer labels.
        known_class_count (int): Number of known devices (closed-set).
        method (str): 'tsne' or 'umap'.
        title (str): Title of the plot.
    """

    assert method in ['tsne', 'umap'], "Method must be 'tsne' or 'umap'"

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)

    reduced = reducer.fit_transform(features)

    # Set up color palette: known classes and open-set
    known_labels = np.unique(labels[labels < known_class_count])
    unknown_mask = labels >= known_class_count

    palette = sns.color_palette("hls", len(known_labels))
    colors = np.array([palette[l] if l < known_class_count else (0.2, 0.2, 0.2) for l in labels])

    plt.figure(figsize=(8, 6))
    for l in known_labels:
        idx = labels == l
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Device {l}", s=15)

    if unknown_mask.any():
        plt.scatter(reduced[unknown_mask, 0], reduced[unknown_mask, 1],
                    label="Unknown", c='black', marker='x', s=30)

    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()
