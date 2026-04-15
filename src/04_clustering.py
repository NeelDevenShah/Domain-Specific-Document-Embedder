"""Cluster embeddings using traditional methods."""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import config
from config import RANDOM_STATE


def find_optimal_k(embeddings: np.ndarray, k_range: range = range(2, 8)) -> int:
    """Pick k with highest silhouette score."""
    best_k, best_score = 2, -1.0
    normed = normalize(embeddings)
    for k in k_range:
        if k >= len(embeddings):
            break
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(normed)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(normed, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def agglomerative_cosine(n_clusters: int) -> AgglomerativeClustering:
    """Create cosine agglomerative clustering across scikit-learn API versions."""
    try:
        return AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=n_clusters, affinity="cosine", linkage="average"
        )


def cluster(
    domain_emb: np.ndarray = None,
    baseline_emb: np.ndarray = None,
    embedding_sets: dict[str, np.ndarray] = None,
    output_dir: Path = None,
    n_clusters: int = None,
) -> dict:
    """
    Run KMeans and Agglomerative clustering on embedding sets.

    Returns cluster labels and chosen k.
    """
    output_dir = output_dir or config.OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)

    if embedding_sets is None:
        if domain_emb is None:
            domain_emb = np.load(output_dir / "domain_embeddings.npy")
        if baseline_emb is None:
            baseline_emb = np.load(output_dir / "baseline_embeddings.npy")
        embedding_sets = {"domain": domain_emb, "baseline": baseline_emb}

    if n_clusters is None:
        first_emb = next(iter(embedding_sets.values()))
        n_clusters = find_optimal_k(first_emb)
    print(f"Using k={n_clusters} clusters")

    results = {}
    for name, emb in embedding_sets.items():
        normed = normalize(emb)
        kmeans_labels = KMeans(
            n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10
        ).fit_predict(normed)
        agg_labels = agglomerative_cosine(n_clusters).fit_predict(normed)

        results[name] = {
            "kmeans_labels": kmeans_labels.tolist(),
            "agglomerative_labels": agg_labels.tolist(),
            "n_clusters": n_clusters,
        }

    cluster_path = output_dir / "cluster_results.json"
    with cluster_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Clustering complete. Results saved to {cluster_path}")
    return results


if __name__ == "__main__":
    cluster()
