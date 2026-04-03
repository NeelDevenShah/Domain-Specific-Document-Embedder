"""Generate plain vs fine-tuned MiniLM embeddings."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import BASELINE_MODEL, DATA_PROCESSED, FINETUNED_MODEL_DIR, OUTPUTS


def load_paragraphs(processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl") -> list[dict]:
    records = []
    with processed_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def generate_minilm_comparison(
    processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl",
    finetuned_model_dir: Path = FINETUNED_MODEL_DIR,
    output_dir: Path = OUTPUTS,
) -> dict:
    """
    Embed corpus with plain (pretrained) and fine-tuned MiniLM.

    Returns embedding arrays and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_paragraphs(processed_path)
    texts = [r["text"] for r in records]

    print(f"Loading plain model: {BASELINE_MODEL}")
    plain_model = SentenceTransformer(BASELINE_MODEL)
    plain_emb = plain_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print(f"Loading fine-tuned model: {finetuned_model_dir}")
    finetuned_model = SentenceTransformer(str(finetuned_model_dir))
    finetuned_emb = finetuned_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(output_dir / "plain_embeddings.npy", plain_emb)
    np.save(output_dir / "finetuned_embeddings.npy", finetuned_emb)

    meta = {
        "ids": [r["id"] for r in records],
        "doc_types": [r["doc_type"] for r in records],
        "source_files": [r["source_file"] for r in records],
        "texts": texts,
        "plain_shape": list(plain_emb.shape),
        "finetuned_shape": list(finetuned_emb.shape),
        "plain_model": BASELINE_MODEL,
        "finetuned_model": str(finetuned_model_dir),
    }
    with (output_dir / "embedding_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Plain embeddings:     {plain_emb.shape}")
    print(f"Fine-tuned embeddings: {finetuned_emb.shape}")
    return {"plain": plain_emb, "finetuned": finetuned_emb, "meta": meta}


if __name__ == "__main__":
    generate_minilm_comparison()
