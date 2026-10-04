import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.rag.vector_store import build_vector_store

store = build_vector_store()
count = store.count()
print(f"Vector Count: {count}")

results = store.search("architecture", top_k=1)
if results:
    payload, score = results[0]
    print("Search Result Found:")
    print("Score:", score)
    print("Keys in payload:", list(payload.keys()))
    print("Citation included in payload:", "citation" in payload or "source" in payload)
else:
    print("No results found.")
