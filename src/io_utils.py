"""Shared file-loading helpers for pipeline stages."""

import json
from pathlib import Path

from config import DATA_PROCESSED


def load_paragraph_records(
    processed_path: Path = DATA_PROCESSED / "paragraphs.jsonl",
) -> list[dict]:
    """Load processed paragraph records from JSONL."""
    records = []
    with processed_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records
