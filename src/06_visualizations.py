"""Generate visualizations comparing embedding sets."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import normalize

import config
from config import RANDOM_STATE


def plot_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
):
    """2D PCA scatter colored by cluster label."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(normalize(embeddings))

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("tab10", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[palette[i]],
            label=f"Cluster {lbl}",
            alpha=0.6,
            s=30,
        )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


DISPLAY_NAMES = {
    "domain": "Domain Word2Vec",
    "baseline": "Baseline MiniLM",
    "plain": "Plain MiniLM",
    "finetuned": "Fine-tuned MiniLM",
}


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def tsne_project(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D with t-SNE across scikit-learn API versions."""
    normed = normalize(embeddings)
    perplexity = min(30, max(2, len(normed) - 1))
    kwargs = {
        "n_components": 2,
        "random_state": RANDOM_STATE,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": 200.0,
    }
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(normed)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(normed)


def plot_tsne_comparison(embedding_sets, cluster_labels, save_path: Path):
    """Side-by-side t-SNE of embedding clusters."""
    n_plots = len(embedding_sets)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7), squeeze=False)
    for ax, (name, emb) in zip(axes[0], embedding_sets.items()):
        labels = cluster_labels[name]
        coords = tsne_project(emb)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=25
        )
        ax.set_title(display_name(name))
        fig.colorbar(scatter, ax=ax, label="Cluster")

    fig.suptitle("t-SNE Cluster Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_silhouette(embeddings, labels, title, save_path: Path):
    """Silhouette plot for cluster quality."""
    normed = normalize(embeddings)
    sample_sil = silhouette_samples(normed, labels, metric="cosine")
    fig, ax = plt.subplots(figsize=(8, 6))

    y_lower = 10
    for i in sorted(set(labels)):
        ith = sample_sil[labels == i]
        ith.sort()
        size = ith.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, ith, alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=np.mean(sample_sil), color="red", linestyle="--", label="Mean")
    ax.set_title(f"Silhouette Plot — {title}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metrics_comparison(metrics: dict, save_path: Path):
    """Bar chart comparing key metrics side-by-side."""
    metric_names = [
        "silhouette_score",
        "adjusted_rand_index",
        "normalized_mutual_info",
    ]
    labels = ["Silhouette", "ARI", "NMI"]
    model_names = [name for name in metrics if name != "nearest_neighbors"]

    x = np.arange(len(labels))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    all_vals = []
    for idx, name in enumerate(model_names):
        vals = [metrics[name][m] for m in metric_names]
        all_vals.extend(vals)
        offset = (idx - (len(model_names) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=display_name(name), color=colors[idx % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Embedding Clustering Metrics")
    ax.legend()
    ax.set_ylim(0, max(all_vals) * 1.2 + 0.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_doc_type_distribution(doc_types: list[str], save_path: Path):
    """Bar chart of paragraph counts per document type."""
    from collections import Counter

    counts = Counter(doc_types)
    fig, ax = plt.subplots(figsize=(8, 5))
    types = list(counts.keys())
    vals = [counts[t] for t in types]
    ax.bar(types, vals, color="#4CAF50")
    ax.set_title("Paragraph Distribution by Document Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_similarity_heatmap(
    embeddings: np.ndarray, sample_size: int, title: str, save_path: Path
):
    """Doc x doc cosine similarity heatmap (sampled)."""
    from sklearn.metrics.pairwise import cosine_similarity

    normed = normalize(embeddings)
    n = min(sample_size, len(normed))
    idx = np.linspace(0, len(normed) - 1, n, dtype=int)
    sim = cosine_similarity(normed[idx])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sim, cmap="RdYlBu_r", center=0, ax=ax, xticklabels=False, yticklabels=False)
    ax.set_title(f"Cosine Similarity Heatmap — {title}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize(output_dir: Path = None, figures_dir: Path = None) -> list[str]:
    """Generate all required visualizations. Returns list of saved figure paths."""
    output_dir = output_dir or config.OUTPUTS
    figures_dir = figures_dir or config.FIGURES
    figures_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "embedding_metadata.json").open() as f:
        meta = json.load(f)
    with (output_dir / "cluster_results.json").open() as f:
        clusters = json.load(f)
    with (output_dir / "metrics.json").open() as f:
        metrics = json.load(f)

    model_names = [name for name in clusters]
    embedding_sets = {
        name: np.load(output_dir / f"{name}_embeddings.npy") for name in model_names
    }
    doc_types = meta["doc_types"]
    cluster_labels = {
        name: np.array(clusters[name]["kmeans_labels"]) for name in model_names
    }

    figure_specs = []
    for idx, name in enumerate(model_names, start=1):
        title = display_name(name)
        figure_specs.append(
            (
                f"{idx:02d}_pca_{name}_clusters.png",
                plot_cluster_scatter,
                (embedding_sets[name], cluster_labels[name], f"{title} — PCA Clusters"),
            )
        )

    figure_specs.append(
        ("03_tsne_comparison.png", plot_tsne_comparison, (embedding_sets, cluster_labels))
    )
    for name in model_names:
        figure_specs.append(
            (
                f"04_silhouette_{name}.png",
                plot_silhouette,
                (embedding_sets[name], cluster_labels[name], display_name(name)),
            )
        )
    figure_specs.extend([
        ("05_metrics_comparison.png", plot_metrics_comparison, (metrics,)),
        ("06_doc_type_distribution.png", plot_doc_type_distribution, (doc_types,)),
    ])

    for idx, name in enumerate(model_names, start=7):
        figure_specs.append(
            (
                f"{idx:02d}_similarity_heatmap_{name}.png",
                plot_similarity_heatmap,
                (embedding_sets[name], 50, display_name(name)),
            )
        )

    saved = []
    for filename, func, args in figure_specs:
        path = figures_dir / filename
        if func in (plot_metrics_comparison, plot_doc_type_distribution):
            func(*args, path)
        elif func == plot_tsne_comparison:
            func(*args, path)
        else:
            func(*args, save_path=path)
        saved.append(str(path))

    print(f"Saved {len(saved)} figures to {figures_dir}")
    return saved


if __name__ == "__main__":
    visualize()
