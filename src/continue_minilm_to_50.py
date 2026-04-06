"""
Continue the 3-epoch MiniLM model to 50 total epochs.

This resumes from outputs/2/models/finetuned_minilm, evaluates against the
plain MiniLM embeddings every 5 total epochs, and writes the continuation run
under outputs/3/.
"""

import csv
import json
import shutil
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_PROCESSED, FINETUNE_BATCH_SIZE, FINETUNE_WARMUP_STEPS

finetune_mod = import_module("02b_finetune_minilm")
clustering_mod = import_module("04_clustering")
evaluation_mod = import_module("05_evaluation_metrics")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_RUN = PROJECT_ROOT / "outputs" / "2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "3"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
MODELS_DIR = OUTPUT_DIR / "models"
START_EPOCH = 3
TARGET_EPOCH = 50
EVAL_EVERY = 5


def load_source_artifacts() -> tuple[np.ndarray, dict]:
    plain_path = SOURCE_RUN / "plain_embeddings.npy"
    metadata_path = SOURCE_RUN / "embedding_metadata.json"

    if not plain_path.exists():
        raise FileNotFoundError(f"Missing plain embeddings: {plain_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    return np.load(plain_path), metadata


def write_metadata(output_dir: Path, metadata: dict, plain_emb: np.ndarray, finetuned_emb: np.ndarray, model_dir: Path):
    checkpoint_meta = {
        **metadata,
        "plain_shape": list(plain_emb.shape),
        "finetuned_shape": list(finetuned_emb.shape),
        "finetuned_model": str(model_dir),
    }
    with (output_dir / "embedding_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(checkpoint_meta, f, indent=2)


def metric_row(total_epoch: int, metrics: dict) -> dict:
    row = {"epoch": total_epoch}
    for model_name in ("plain", "finetuned"):
        model_metrics = metrics[model_name]
        prefix = f"{model_name}_"
        row[f"{prefix}silhouette_score"] = model_metrics["silhouette_score"]
        row[f"{prefix}davies_bouldin_index"] = model_metrics["davies_bouldin_index"]
        row[f"{prefix}calinski_harabasz_score"] = model_metrics["calinski_harabasz_score"]
        row[f"{prefix}adjusted_rand_index"] = model_metrics["adjusted_rand_index"]
        row[f"{prefix}normalized_mutual_info"] = model_metrics["normalized_mutual_info"]
        row[f"{prefix}separation_gap"] = model_metrics["similarity"]["separation_gap"]

    row["delta_silhouette_score"] = row["finetuned_silhouette_score"] - row["plain_silhouette_score"]
    row["delta_davies_bouldin_index"] = row["finetuned_davies_bouldin_index"] - row["plain_davies_bouldin_index"]
    row["delta_calinski_harabasz_score"] = (
        row["finetuned_calinski_harabasz_score"] - row["plain_calinski_harabasz_score"]
    )
    row["delta_adjusted_rand_index"] = row["finetuned_adjusted_rand_index"] - row["plain_adjusted_rand_index"]
    row["delta_normalized_mutual_info"] = (
        row["finetuned_normalized_mutual_info"] - row["plain_normalized_mutual_info"]
    )
    row["delta_separation_gap"] = row["finetuned_separation_gap"] - row["plain_separation_gap"]
    return row


def save_log(rows: list[dict]):
    json_path = OUTPUT_DIR / "metrics_log.json"
    csv_path = OUTPUT_DIR / "metrics_log.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if not rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def completed_epochs() -> set[int]:
    if not CHECKPOINTS_DIR.exists():
        return set()
    return {
        int(path.name.removeprefix("epoch_"))
        for path in CHECKPOINTS_DIR.glob("epoch_*")
        if path.name.removeprefix("epoch_").isdigit() and (path / "metrics.json").exists()
    }


def evaluate_checkpoint(
    model: SentenceTransformer,
    total_epoch: int,
    texts: list[str],
    plain_emb: np.ndarray,
    metadata: dict,
) -> dict:
    checkpoint_dir = CHECKPOINTS_DIR / f"epoch_{total_epoch:02d}"
    model_dir = MODELS_DIR / f"finetuned_minilm_epoch_{total_epoch:02d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating total epoch {total_epoch}...")
    model.save(str(model_dir))
    finetuned_emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(checkpoint_dir / "plain_embeddings.npy", plain_emb)
    np.save(checkpoint_dir / "finetuned_embeddings.npy", finetuned_emb)
    write_metadata(checkpoint_dir, metadata, plain_emb, finetuned_emb, model_dir)

    embedding_sets = {"plain": plain_emb, "finetuned": finetuned_emb}
    clustering_mod.cluster(embedding_sets=embedding_sets, output_dir=checkpoint_dir)
    metrics = evaluation_mod.evaluate(embedding_sets=embedding_sets, output_dir=checkpoint_dir)
    print(f"Epoch {total_epoch} metrics saved to {checkpoint_dir / 'metrics.json'}")
    return metrics


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    source_model = SOURCE_RUN / "models" / "finetuned_minilm"
    if not source_model.exists():
        raise FileNotFoundError(f"Missing 3-epoch source model: {source_model}")

    plain_emb, metadata = load_source_artifacts()
    texts = metadata["texts"]
    records = finetune_mod.load_paragraphs(DATA_PROCESSED / "paragraphs.jsonl")
    train_examples = finetune_mod.build_training_pairs(records)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=FINETUNE_BATCH_SIZE)

    done = completed_epochs()
    rows = []
    if (OUTPUT_DIR / "metrics_log.json").exists():
        with (OUTPUT_DIR / "metrics_log.json").open(encoding="utf-8") as f:
            rows = json.load(f)

    print("=" * 60)
    print("Continue MiniLM fine-tuning: epoch 3 -> epoch 50")
    print("=" * 60)
    print(f"Source model: {source_model}")
    print(f"Training pairs: {len(train_examples)}")
    print(f"Already evaluated checkpoints: {sorted(done)}")

    model = SentenceTransformer(str(source_model))
    current_epoch = START_EPOCH

    for target_epoch in range(EVAL_EVERY, TARGET_EPOCH + 1, EVAL_EVERY):
        if target_epoch <= START_EPOCH:
            continue
        if target_epoch in done:
            current_epoch = target_epoch
            latest_model = MODELS_DIR / f"finetuned_minilm_epoch_{target_epoch:02d}"
            if latest_model.exists():
                model = SentenceTransformer(str(latest_model))
            continue

        epochs_to_train = target_epoch - current_epoch
        print(f"\nTraining from epoch {current_epoch} to {target_epoch} ({epochs_to_train} epochs)...")
        train_loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs_to_train,
            warmup_steps=FINETUNE_WARMUP_STEPS,
            show_progress_bar=True,
        )
        current_epoch = target_epoch

        metrics = evaluate_checkpoint(model, target_epoch, texts, plain_emb, metadata)
        rows = [row for row in rows if row["epoch"] != target_epoch]
        rows.append(metric_row(target_epoch, metrics))
        rows.sort(key=lambda row: row["epoch"])
        save_log(rows)

    final_model = MODELS_DIR / "finetuned_minilm_epoch_50"
    if final_model.exists():
        shutil.copytree(final_model, OUTPUT_DIR / "finetuned_minilm_final", dirs_exist_ok=True)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Continuation complete in {elapsed:.1f}s")
    print(f"Metrics log: {OUTPUT_DIR / 'metrics_log.json'}")
    print(f"CSV log:     {OUTPUT_DIR / 'metrics_log.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
