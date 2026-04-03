"""Fine-tune all-MiniLM-L6-v2 on the legal corpus."""

import json
import random
from collections import defaultdict
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from config import (
    BASELINE_MODEL,
    DATA_PROCESSED,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_MAX_PAIRS_PER_DOC,
    FINETUNE_MODEL_DIR,
    FINETUNE_WARMUP_STEPS,
    RANDOM_STATE,
)


def load_paragraphs(processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl") -> list[dict]:
    records = []
    with processed_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_training_pairs(records: list[dict], max_pairs_per_doc: int = FINETUNE_MAX_PAIRS_PER_DOC) -> list[InputExample]:
    """
    Build anchor-positive pairs from paragraphs in the same source document.

    Same-document pairs teach the model that legal paragraphs from one filing
    share semantic space, while in-batch negatives provide contrast.
    """
    random.seed(RANDOM_STATE)
    by_source: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_source[record["source_file"]].append(record["text"])

    examples = []
    for texts in by_source.values():
        if len(texts) < 2:
            continue
        pairs = [(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        random.shuffle(pairs)
        for a, b in pairs[:max_pairs_per_doc]:
            examples.append(InputExample(texts=[a, b]))

    random.shuffle(examples)
    return examples


def finetune_minilm(
    processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl",
    output_dir: Path = FINETUNED_MODEL_DIR,
    base_model: str = BASELINE_MODEL,
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = FINETUNE_BATCH_SIZE,
) -> Path:
    """
    Fine-tune MiniLM on legal paragraph pairs using MultipleNegativesRankingLoss.

    Returns path to saved fine-tuned model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_paragraphs(processed_path)
    train_examples = build_training_pairs(records)

    print(f"Fine-tuning {base_model} on {len(train_examples)} same-document pairs")
    print(f"  epochs={epochs}, batch_size={batch_size}, warmup={FINETUNE_WARMUP_STEPS}")

    model = SentenceTransformer(base_model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=FINETUNE_WARMUP_STEPS,
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    print(f"Fine-tuned model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    finetune_minilm()
