"""
End-to-end pipeline for plain vs fine-tuned MiniLM comparison.

Usage:
    python src/main_minilm_comparison.py
"""

import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

finetune_mod = import_module("02b_finetune_minilm")
generate_mod = import_module("03b_generate_minilm_comparison")
clustering_mod = import_module("04_clustering")
evaluation_mod = import_module("05_evaluation_metrics")
visualizations_mod = import_module("06_visualizations")


def main():
    start = time.time()
    print("=" * 60)
    print("Plain vs Fine-tuned MiniLM — Full Pipeline")
    print("=" * 60)

    print("\n[1/5] Fine-tuning MiniLM...")
    finetune_mod.finetune_minilm()

    print("\n[2/5] Generating plain and fine-tuned embeddings...")
    emb_result = generate_mod.generate_minilm_comparison()
    embedding_sets = {
        "plain": emb_result["plain"],
        "finetuned": emb_result["finetuned"],
    }

    print("\n[3/5] Clustering...")
    clustering_mod.cluster(embedding_sets=embedding_sets)

    print("\n[4/5] Evaluating...")
    metrics = evaluation_mod.evaluate(embedding_sets=embedding_sets)

    print("\n[5/5] Generating visualizations...")
    visualizations_mod.visualize()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print("  Comparison: plain all-MiniLM-L6-v2 vs fine-tuned all-MiniLM-L6-v2")
    print("  Outputs saved to: outputs/2/")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
