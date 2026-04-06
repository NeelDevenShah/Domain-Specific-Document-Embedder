"""Visualize how fine-tuning duration (5-50 epochs) affects clustering metrics.

Reads outputs/3/metrics_log.csv (produced by continue_minilm_to_50.py) and plots
the finetuned-model metrics against epoch count, plus figures for the final
(epoch 50) checkpoint reusing the standard visualization helpers.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "outputs" / "3"
FIGURES_DIR = RUN_DIR / "figures"
FINAL_CHECKPOINT = RUN_DIR / "checkpoints" / "epoch_50"


def load_log(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_epoch_trend(rows: list[dict], save_path: Path):
    epochs = [int(r["epoch"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(epochs, [float(r["finetuned_adjusted_rand_index"]) for r in rows], "o-", label="ARI", color="#2196F3")
    ax.plot(epochs, [float(r["finetuned_normalized_mutual_info"]) for r in rows], "o-", label="NMI", color="#4CAF50")
    ax.plot(epochs, [float(r["finetuned_separation_gap"]) for r in rows], "o-", label="Separation gap", color="#9C27B0")
    ax.axhline(float(rows[0]["plain_adjusted_rand_index"]), color="#2196F3", linestyle=":", alpha=0.6, label="Plain ARI")
    ax.axhline(float(rows[0]["plain_normalized_mutual_info"]), color="#4CAF50", linestyle=":", alpha=0.6, label="Plain NMI")
    ax.set_xlabel("Fine-tuning epochs")
    ax.set_ylabel("Score")
    ax.set_title("Label-alignment metrics improve with more fine-tuning")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(epochs, [float(r["finetuned_silhouette_score"]) for r in rows], "o-", label="Silhouette", color="#FF9800")
    ax2b = ax2.twinx()
    ax2b.plot(epochs, [float(r["finetuned_calinski_harabasz_score"]) for r in rows], "s--", label="Calinski-Harabasz", color="#F44336")
    ax2.axhline(float(rows[0]["plain_silhouette_score"]), color="#FF9800", linestyle=":", alpha=0.6, label="Plain silhouette")
    ax2.set_xlabel("Fine-tuning epochs")
    ax2.set_ylabel("Silhouette")
    ax2b.set_ylabel("Calinski-Harabasz")
    ax2.set_title("Internal cluster geometry degrades past ~epoch 10")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    fig.suptitle("Fine-tuning duration ablation: label-alignment vs. cluster geometry tradeoff", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_log(RUN_DIR / "metrics_log.csv")
    out_path = FIGURES_DIR / "01_epoch_trend.png"
    plot_epoch_trend(rows, out_path)
    print(f"Saved {out_path}")

    if FINAL_CHECKPOINT.exists():
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from importlib import import_module

        viz_mod = import_module("06_visualizations")
        saved = viz_mod.visualize(output_dir=FINAL_CHECKPOINT, figures_dir=FIGURES_DIR)
        print(f"Saved {len(saved)} epoch-50 checkpoint figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
