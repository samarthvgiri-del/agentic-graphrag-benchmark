"""
CLI ingestion script.

Usage:
    python scripts/ingest.py <path>         # Ingest a single file
    python scripts/ingest.py <directory>    # Ingest all supported files in directory

Must be run from the backend/ directory (or with backend/ on PYTHONPATH).
"""

import sys
import os
import json

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.vector_store import build_vector_store
from app.rag.ingest_service import ingest_file, ingest_directory


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <file_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    store = build_vector_store()

    if os.path.isdir(target):
        results = ingest_directory(target, store)
    elif os.path.isfile(target):
        results = [ingest_file(target, store)]
    else:
        print(f"Error: '{target}' not found")
        sys.exit(1)

    print(json.dumps(results, indent=2))
    print(f"\nQdrant status: {json.dumps(store.get_status(), indent=2)}")


if __name__ == "__main__":
    main()
