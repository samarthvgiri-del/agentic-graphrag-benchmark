"""
Document loaders for supported file formats (Phase 3).

Each loader returns a list of dicts with keys:
  - text: str  (the extracted text content)
  - source: str  (original filename)
  - page: Optional[int]  (page number for PDFs, None otherwise)
"""

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Supported extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".csv", ".jsonl"}


def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Route to the correct loader based on file extension.

    Returns a list of raw page/section dicts before chunking.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    loader = _LOADERS[ext]
    pages = loader(path)
    logger.info("Loaded %d section(s) from %s", len(pages), path.name)
    return pages


# ------------------------------------------------------------------
# Individual loaders
# ------------------------------------------------------------------

def _load_txt(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"text": text, "source": path.name, "page": None}]


def _load_md(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"text": text, "source": path.name, "page": None}]


def _load_json(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Flatten: if it's a list of objects, each becomes a section
    if isinstance(raw, list):
        sections = []
        for idx, item in enumerate(raw):
            text = json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
            sections.append({"text": text, "source": path.name, "page": idx})
        return sections
    # Single object or string
    text = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
    return [{"text": text, "source": path.name, "page": None}]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    sections = []
    text_content = path.read_text(encoding="utf-8")
    for idx, line in enumerate(text_content.strip().split('\n')):
        if not line.strip(): continue
        raw = json.loads(line)
        text = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
        sections.append({"text": text, "source": path.name, "page": idx})
    return sections


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(StringIO(text))
    rows = list(reader)
    if not rows:
        return []
    # Combine header + rows into readable text blocks (one per row)
    header = rows[0]
    sections = []
    for idx, row in enumerate(rows[1:], start=1):
        line = " | ".join(f"{h}: {v}" for h, v in zip(header, row))
        sections.append({"text": line, "source": path.name, "page": idx})
    return sections


def _load_pdf(path: Path) -> List[Dict[str, Any]]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. "
            "Install it with: pip install pymupdf"
        )
    doc = fitz.open(str(path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "source": path.name,
                "page": page_num + 1,
            })
    doc.close()
    return pages


_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".md": _load_md,
    ".json": _load_json,
    ".csv": _load_csv,
    ".jsonl": _load_jsonl,
}
