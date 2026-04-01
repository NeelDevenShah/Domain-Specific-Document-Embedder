"""Generate visualizations comparing domain vs baseline embeddings."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import normalize

from config import FIGURES, OUTPUTS, RANDOM_STATE


def plot_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    doc_types: list[str],
    title: str,
    save_path: Path,
):
    """2D PCA scatter colored by cluster, with doc-type markers."""
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


def plot_tsne_comparison(
    domain_emb, baseline_emb, domain_labels, baseline_labels, save_path: Path
):
    """Side-by-side t-SNE of domain vs baseline clusters."""
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=1000)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, emb, labels, title in zip(
        axes,
        [domain_emb, baseline_emb],
        [domain_labels, baseline_labels],
        ["Domain Word2Vec", "Baseline MiniLM"],
    ):
        coords = tsne.fit_transform(normalize(emb))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=25
        )
        ax.set_title(title)
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
    domain_vals = [metrics["domain"][m] for m in metric_names]
    baseline_vals = [metrics["baseline"][m] for m in metric_names]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, domain_vals, width, label="Domain Word2Vec", color="#2196F3")
    ax.bar(x + width / 2, baseline_vals, width, label="Baseline MiniLM", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Domain vs Baseline — Clustering Metrics")
    ax.legend()
    ax.set_ylim(0, max(max(domain_vals), max(baseline_vals)) * 1.2 + 0.05)
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


def visualize(output_dir: Path = OUTPUTS, figures_dir: Path = FIGURES) -> list[str]:
    """Generate all required visualizations. Returns list of saved figure paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    domain_emb = np.load(output_dir / "domain_embeddings.npy")
    baseline_emb = np.load(output_dir / "baseline_embeddings.npy")

    with (output_dir / "embedding_metadata.json").open() as f:
        meta = json.load(f)
    with (output_dir / "cluster_results.json").open() as f:
        clusters = json.load(f)
    with (output_dir / "metrics.json").open() as f:
        metrics = json.load(f)

    doc_types = meta["doc_types"]
    domain_labels = np.array(clusters["domain"]["kmeans_labels"])
    baseline_labels = np.array(clusters["baseline"]["kmeans_labels"])

    figure_specs = [
        ("01_pca_domain_clusters.png", plot_cluster_scatter, (domain_emb, domain_labels, doc_types, "Domain Word2Vec — PCA Clusters")),
        ("02_pca_baseline_clusters.png", plot_cluster_scatter, (baseline_emb, baseline_labels, doc_types, "Baseline MiniLM — PCA Clusters")),
        ("03_tsne_comparison.png", plot_tsne_comparison, (domain_emb, baseline_emb, domain_labels, baseline_labels)),
        ("04_silhouette_domain.png", plot_silhouette, (domain_emb, domain_labels, "Domain Word2Vec")),
        ("05_metrics_comparison.png", plot_metrics_comparison, (metrics,)),
        ("06_doc_type_distribution.png", plot_doc_type_distribution, (doc_types,)),
        ("07_similarity_heatmap_domain.png", plot_similarity_heatmap, (domain_emb, 50, "Domain Word2Vec")),
        ("08_similarity_heatmap_baseline.png", plot_similarity_heatmap, (baseline_emb, 50, "Baseline MiniLM")),
    ]

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
