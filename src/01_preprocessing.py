"""Extract text from legal PDFs and split into paragraph-level units."""

import json
import re
from pathlib import Path

import pdfplumber

from config import (
    DATA_PROCESSED,
    DATA_RAW,
    DOC_TYPE_LABELS,
    MIN_PARAGRAPH_CHARS,
)

LEGAL_BOILERPLATE = re.compile(
    r"^\s*(?:Page \d+ of \d+|PageID #:\S*|No\. \d+-\d+\s*$|"
    r"IN THE (?:SUPREME|UNITED STATES) COURT.*$|"
    r"Counsel of Record.*$|"
    r"\(\s*Slip Opinion\s*\).*$)",
    re.IGNORECASE | re.MULTILINE,
)

PAGE_NUMBER = re.compile(r"\bPage \d+ of \d+\b", re.IGNORECASE)
MULTI_SPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    chunks = []
    with pdfplumber.open(pdf_path) as doc:
        for page in doc.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def clean_legal_text(text: str) -> str:
    """Clean legal text while preserving domain vocabulary."""
    text = text.replace("\r", "\n")
    text = PAGE_NUMBER.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    text = MULTI_SPACE.sub(" ", text)
    return text.strip()


def split_paragraphs(text: str, min_chars: int = MIN_PARAGRAPH_CHARS) -> list[str]:
    """Split document into paragraph-level units for embedding."""
    # Try double-newline split first
    raw_parts = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_parts if len(p.strip()) >= min_chars]

    # PDF extraction often lacks blank lines — fall back to line-based splitting
    if len(paragraphs) < 5:
        lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_chars]
        paragraphs = lines if lines else paragraphs

    # Still too few units — chunk by sentence groups (~600 chars)
    if len(paragraphs) < 10:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        target_size = 600
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) > target_size and current:
                chunks.append(current.strip())
                current = sent
            else:
                current = f"{current} {sent}".strip()
        if current and len(current) >= min_chars:
            chunks.append(current.strip())
        if chunks:
            paragraphs = chunks

    # Filter boilerplate
    filtered = []
    for part in paragraphs:
        if LEGAL_BOILERPLATE.match(part):
            continue
        filtered.append(part)
    return filtered if filtered else paragraphs


def preprocess(raw_dir: Path = DATA_RAW, output_dir: Path = DATA_PROCESSED) -> dict:
    """
    Extract, clean, and segment all PDFs into paragraph records.

    Returns metadata dict with corpus statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_chars = 0

    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        doc_type = DOC_TYPE_LABELS.get(pdf_path.name, "unknown")
        raw_text = extract_pdf_text(pdf_path)
        cleaned = clean_legal_text(raw_text)
        paragraphs = split_paragraphs(cleaned)

        for idx, paragraph in enumerate(paragraphs):
            records.append(
                {
                    "id": f"{pdf_path.stem}__p{idx:04d}",
                    "source_file": pdf_path.name,
                    "doc_type": doc_type,
                    "paragraph_index": idx,
                    "text": paragraph,
                }
            )
            total_chars += len(paragraph)

    processed_path = output_dir / "paragraphs.jsonl"
    with processed_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    vocab = set()
    for record in records:
        vocab.update(record["text"].lower().split())

    stats = {
        "num_documents": len(list(raw_dir.glob("*.pdf"))),
        "num_paragraphs": len(records),
        "avg_paragraph_length": total_chars / max(len(records), 1),
        "vocab_size": len(vocab),
        "doc_type_counts": {},
        "output_path": str(processed_path),
    }
    for record in records:
        stats["doc_type_counts"][record["doc_type"]] = (
            stats["doc_type_counts"].get(record["doc_type"], 0) + 1
        )

    stats_path = output_dir / "preprocessing_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Preprocessed {stats['num_documents']} PDFs -> {stats['num_paragraphs']} paragraphs")
    print(f"Vocab size: {stats['vocab_size']}, avg paragraph length: {stats['avg_paragraph_length']:.0f} chars")
    return stats


if __name__ == "__main__":
    preprocess()
