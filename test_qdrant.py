import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.rag.vector_store import build_vector_store

try:
    store = build_vector_store()
    print("Qdrant Connection Status:")
    print(store.get_status())
    print("Vector Count:", store.count())
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
