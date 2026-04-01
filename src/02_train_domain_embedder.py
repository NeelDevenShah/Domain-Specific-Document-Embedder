"""Train a domain-specific Word2Vec embedder on the legal corpus."""

import json
from pathlib import Path

from gensim.models import Word2Vec

from config import DATA_PROCESSED, EMBEDDING_DIM, MODELS, RANDOM_STATE


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer preserving legal tokens."""
    return text.lower().split()


def load_corpus(processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl") -> list[list[str]]:
    """Load tokenized sentences from processed paragraphs."""
    sentences = []
    with processed_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            sentences.append(tokenize(record["text"]))
    return sentences


def train_embedder(
    processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl",
    model_dir: Path = MODELS,
    vector_size: int = EMBEDDING_DIM,
) -> Path:
    """
    Train Word2Vec on domain corpus and save the model.

    Returns path to saved model.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_corpus(processed_path)

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=4,
        sg=1,  # skip-gram
        epochs=20,
        seed=RANDOM_STATE,
    )

    model_path = model_dir / "domain_word2vec.model"
    model.save(str(model_path))

    print(f"Trained Word2Vec: vocab={len(model.wv)}, dim={vector_size}, epochs=20")
    print(f"Saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train_embedder()
