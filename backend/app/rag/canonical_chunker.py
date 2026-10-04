import json
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    source_filename: str
    source_line_number: Optional[int]
    chunk_index: int
    start_offset: int
    end_offset: int
    text: str

def get_canonical_chunks(
    jsonl_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[ChunkRecord]:
    """
    Generates canonical chunks identical to the original Qdrant ingestion.
    """
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    lines = text_content.strip().split('\n')

    for idx, line in enumerate(lines):
        if not line.strip(): continue

        raw = json.loads(line)
        text = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw

        # Normalise whitespace as done in chunker.py
        text = " ".join(text.split())
        if not text:
            continue

        # original Qdrant source was just the filename "corpus.jsonl"
        source = "corpus.jsonl"
        page_num = idx

        # Actual doc ID
        document_id = str(raw.get("doc_id") or raw.get("id") or f"doc_{idx}")

        start = 0
        local_idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Exact formula from previous chunker.py
            raw_id = f"{source}::{page_num}::{start}::{chunk_text[:50]}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:12]

            rec = ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                source_filename=source,
                source_line_number=idx,
                chunk_index=local_idx,
                start_offset=start,
                end_offset=end,
                text=chunk_text
            )
            records.append(rec)

            local_idx += 1
            start += chunk_size - chunk_overlap

    return records
