"""Generate domain and baseline embeddings for all paragraph units."""

import json
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

import config
from config import BASELINE_MODEL, DATA_PROCESSED
from io_utils import load_paragraph_records


def domain_paragraph_embedding(text: str, model: Word2Vec) -> np.ndarray:
    """Average Word2Vec vectors for tokens present in the model vocabulary."""
    tokens = text.lower().split()
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def generate_embeddings(
    processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl",
    model_path: Path = None,
    output_dir: Path = None,
) -> dict:
    """
    Generate domain (Word2Vec) and baseline (MiniLM) embeddings.

    Returns paths and metadata.
    """
    model_path = model_path or config.MODELS / "domain_word2vec.model"
    output_dir = output_dir or config.OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_paragraph_records(processed_path)
    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    doc_types = [r["doc_type"] for r in records]

    # Domain embeddings
    w2v = Word2Vec.load(str(model_path))
    domain_emb = np.vstack([domain_paragraph_embedding(t, w2v) for t in texts])

    # Baseline embeddings
    print(f"Loading baseline model: {BASELINE_MODEL}")
    baseline_model = SentenceTransformer(BASELINE_MODEL)
    baseline_emb = baseline_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings
    np.save(output_dir / "domain_embeddings.npy", domain_emb)
    np.save(output_dir / "baseline_embeddings.npy", baseline_emb)

    meta = {
        "ids": ids,
        "doc_types": doc_types,
        "texts": texts,
        "domain_shape": list(domain_emb.shape),
        "baseline_shape": list(baseline_emb.shape),
    }
    meta_path = output_dir / "embedding_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Domain embeddings: {domain_emb.shape}")
    print(f"Baseline embeddings: {baseline_emb.shape}")
    return {"domain": domain_emb, "baseline": baseline_emb, "meta": meta}


if __name__ == "__main__":
    generate_embeddings()
