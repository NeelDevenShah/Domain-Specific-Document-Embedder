"""
End-to-end pipeline for domain-specific legal document embedder.

Usage (with conda base):
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate base
    cd "/home/neel/Desktop/Custom Embedding"
    python src/main.py
"""

import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
# Update it based on the experiment number that you are performing
config.set_run_id("1")

preprocessing = import_module("01_preprocessing")
train_embedder_mod = import_module("02_train_domain_embedder")
generate_embeddings_mod = import_module("03_generate_embeddings")
clustering_mod = import_module("04_clustering")
evaluation_mod = import_module("05_evaluation_metrics")
visualizations_mod = import_module("06_visualizations")


def main():
    start = time.time()
    print("=" * 60)
    print("Domain-Specific Legal Embedder — Full Pipeline")
    print("=" * 60)

    # Step 1: Preprocess PDFs
    print("\n[1/6] Preprocessing PDFs...")
    stats = preprocessing.preprocess()

    # Step 2: Train domain embedder
    print("\n[2/6] Training domain Word2Vec embedder...")
    train_embedder_mod.train_embedder()

    # Step 3: Generate embeddings (domain + baseline)
    print("\n[3/6] Generating embeddings...")
    emb_result = generate_embeddings_mod.generate_embeddings()

    # Step 4: Clustering
    print("\n[4/6] Clustering...")
    clustering_mod.cluster(
        domain_emb=emb_result["domain"],
        baseline_emb=emb_result["baseline"],
    )

    # Step 5: Evaluation
    print("\n[5/6] Evaluating...")
    metrics = evaluation_mod.evaluate(
        domain_emb=emb_result["domain"],
        baseline_emb=emb_result["baseline"],
    )

    # Step 6: Visualizations
    print("\n[6/6] Generating visualizations...")
    visualizations_mod.visualize()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Paragraphs processed: {stats['num_paragraphs']}")
    print(f"  Figures saved to: {config.FIGURES}/")
    print(f"  Metrics saved to: {config.OUTPUTS / 'metrics.json'}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
