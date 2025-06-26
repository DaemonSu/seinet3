import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.cm as cm



def visualize_features(features, labels, known_class_count, method='tsne', prototypes=None, proto_threshold=None):
    all_data = features
    if prototypes is not None:
        all_data = np.concatenate([features, prototypes], axis=0)

    embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_data)

    feat_emb = embedded[:len(features)]
    if prototypes is not None:
        proto_emb = embedded[len(features):]

    plt.figure(figsize=(10, 8))

    labels = np.array(labels)
    cmap = cm.get_cmap('tab10', known_class_count)

    # 绘制已知类别
    for cls in range(known_class_count):
        mask = labels == cls
        plt.scatter(feat_emb[mask, 0], feat_emb[mask, 1], label=f"Class {cls}", color=cmap(cls), s=20, alpha=0.6)

    # 绘制开集样本
    open_mask = labels == -1
    if open_mask.sum() > 0:
        plt.scatter(feat_emb[open_mask, 0], feat_emb[open_mask, 1], label="Unknown", color='gray', s=20, alpha=0.6, marker='x')

    # 绘制原型
    if prototypes is not None:
        for i in range(min(known_class_count, len(proto_emb))):
            px, py = proto_emb[i]
            plt.scatter(px, py, marker='X', s=200, color=cmap(i), edgecolors='black', linewidths=1.5, label=f"Proto {i}")

            # 绘制距离阈值圆
            if proto_threshold is not None:
                circle = Circle((px, py), proto_threshold, color=cmap(i), linestyle='--', fill=False, alpha=0.4)
                plt.gca().add_patch(circle)

    plt.legend(loc='best')
    plt.title(f"{method.upper()} Feature Visualization (Known + Unknown)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# def visualize_features(features, labels, known_class_count, method='tsne', title=''):
#     """
#     Visualize feature embeddings using t-SNE or UMAP.
#
#     Args:
#         features (np.ndarray): Shape (N, D) feature matrix.
#         labels (np.ndarray): Shape (N,) integer labels.
#         known_class_count (int): Number of known devices (closed-set).
#         method (str): 'tsne' or 'umap'.
#         title (str): Title of the plot.
#     """
#
#     assert method in ['tsne', 'umap'], "Method must be 'tsne' or 'umap'"
#
#     if method == 'tsne':
#         reducer = TSNE(n_components=2, perplexity=30, random_state=42)
#     else:
#         reducer = umap.UMAP(n_components=2, random_state=42)
#
#     reduced = reducer.fit_transform(features)
#
#     # Set up color palette: known classes and open-set
#     known_labels = np.unique(labels[labels < known_class_count])
#     unknown_mask = labels >= known_class_count
#
#     palette = sns.color_palette("hls", len(known_labels))
#     colors = np.array([palette[l] if l < known_class_count else (0.2, 0.2, 0.2) for l in labels])
#
#     plt.figure(figsize=(8, 6))
#     for l in known_labels:
#         idx = labels == l
#         plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Device {l}", s=15)
#
#     if unknown_mask.any():
#         plt.scatter(reduced[unknown_mask, 0], reduced[unknown_mask, 1],
#                     label="Unknown", c='black', marker='x', s=30)
#
#     plt.legend()
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()
