"""Fine-tune all-MiniLM-L6-v2 on the legal corpus."""

import random
from collections import defaultdict
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

import config
from config import (
    BASELINE_MODEL,
    DATA_PROCESSED,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_MAX_PAIRS_PER_DOC,
    FINETUNE_WARMUP_STEPS,
    RANDOM_STATE,
)
from io_utils import load_paragraph_records


load_paragraphs = load_paragraph_records


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
        max_possible = len(texts) * (len(texts) - 1) // 2
        target_pairs = min(max_pairs_per_doc, max_possible)
        seen_pairs: set[tuple[int, int]] = set()

        while len(seen_pairs) < target_pairs:
            i, j = sorted(random.sample(range(len(texts)), 2))
            if (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))
            examples.append(InputExample(texts=[texts[i], texts[j]]))

    random.shuffle(examples)
    return examples


def finetune_minilm(
    processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl",
    output_dir: Path = None,
    base_model: str = BASELINE_MODEL,
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = FINETUNE_BATCH_SIZE,
) -> Path:
    """
    Fine-tune MiniLM on legal paragraph pairs using MultipleNegativesRankingLoss.

    Returns path to saved fine-tuned model.
    """
    output_dir = output_dir or config.FINETUNED_MODEL_DIR
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
    config.set_run_id("2")
    finetune_minilm()
