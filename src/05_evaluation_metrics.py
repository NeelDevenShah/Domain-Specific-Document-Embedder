"""Evaluate embedding quality via clustering and similarity metrics."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from config import OUTPUTS


def intra_inter_similarity(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute mean intra-cluster vs inter-cluster cosine similarity."""
    sim = cosine_similarity(normalize(embeddings))
    intra_sims, inter_sims = [], []

    unique_labels = set(labels)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_sims.append(sim[i, j])
            else:
                inter_sims.append(sim[i, j])

    return {
        "intra_cluster_mean": float(np.mean(intra_sims)) if intra_sims else 0.0,
        "inter_cluster_mean": float(np.mean(inter_sims)) if inter_sims else 0.0,
        "separation_gap": float(
            (np.mean(intra_sims) if intra_sims else 0) - (np.mean(inter_sims) if inter_sims else 0)
        ),
    }


def nearest_neighbors(
    embeddings: np.ndarray,
    texts: list[str],
    ids: list[str],
    sample_indices: list[int],
    k: int = 3,
) -> list[dict]:
    """Return top-k nearest neighbors for sample documents."""
    sim = cosine_similarity(normalize(embeddings))
    results = []
    for idx in sample_indices:
        scores = sim[idx].copy()
        scores[idx] = -1  # exclude self
        top_k = np.argsort(scores)[::-1][:k]
        results.append(
            {
                "query_id": ids[idx],
                "query_text": texts[idx][:200] + "...",
                "neighbors": [
                    {
                        "id": ids[j],
                        "similarity": float(scores[j]),
                        "text_preview": texts[j][:150] + "...",
                    }
                    for j in top_k
                ],
            }
        )
    return results


def evaluate(
    domain_emb: np.ndarray = None,
    baseline_emb: np.ndarray = None,
    embedding_sets: dict[str, np.ndarray] = None,
    output_dir: Path = OUTPUTS,
) -> dict:
    """
    Compute 5+ metrics comparing embedding sets.

    Returns metrics dict saved to outputs/metrics.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if embedding_sets is None:
        if domain_emb is None:
            domain_emb = np.load(output_dir / "domain_embeddings.npy")
        if baseline_emb is None:
            baseline_emb = np.load(output_dir / "baseline_embeddings.npy")
        embedding_sets = {"domain": domain_emb, "baseline": baseline_emb}

    with (output_dir / "embedding_metadata.json").open() as f:
        meta = json.load(f)
    with (output_dir / "cluster_results.json").open() as f:
        clusters = json.load(f)

    true_labels = meta["doc_types"]
    texts = meta["texts"]
    ids = meta["ids"]

    # Encode true labels as integers for ARI/NMI
    label_map = {t: i for i, t in enumerate(sorted(set(true_labels)))}
    true_int = np.array([label_map[t] for t in true_labels])

    metrics = {}
    for name, emb in embedding_sets.items():
        labels = np.array(clusters[name]["kmeans_labels"])
        normed = normalize(emb)

        metrics[name] = {
            "silhouette_score": float(silhouette_score(normed, labels, metric="cosine")),
            "davies_bouldin_index": float(davies_bouldin_score(normed, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(normed, labels)),
            "adjusted_rand_index": float(adjusted_rand_score(true_int, labels)),
            "normalized_mutual_info": float(
                normalized_mutual_info_score(true_int, labels, average_method="arithmetic")
            ),
            "similarity": intra_inter_similarity(emb, labels),
        }

    # Nearest-neighbor sanity checks (3 samples per embedder)
    sample_indices = [0, len(texts) // 3, 2 * len(texts) // 3]
    metrics["nearest_neighbors"] = {
        name: nearest_neighbors(emb, texts, ids, sample_indices)
        for name, emb in embedding_sets.items()
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Metrics ===")
    for name in embedding_sets:
        m = metrics[name]
        print(f"\n{name.upper()}:")
        print(f"  Silhouette:        {m['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin:    {m['davies_bouldin_index']:.4f} (lower=better)")
        print(f"  Calinski-Harabasz: {m['calinski_harabasz_score']:.2f}")
        print(f"  ARI (vs doc type): {m['adjusted_rand_index']:.4f}")
        print(f"  NMI (vs doc type): {m['normalized_mutual_info']:.4f}")
        print(f"  Intra/Inter gap:   {m['similarity']['separation_gap']:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate()
