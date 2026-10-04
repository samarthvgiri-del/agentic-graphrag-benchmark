import httpx
import asyncio
import json

async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        # First check ingestion status
        status = await c.get("http://127.0.0.1:8000/api/ingest/status")
        print("=== Qdrant Status ===")
        print(json.dumps(status.json(), indent=2))

        # Run a real RAG query through Qdrant
        print("\n=== RAG Query (via Qdrant) ===")
        r = await c.post(
            "http://127.0.0.1:8000/api/query/rag",
            json={"question": "What is TigerGraph and how is it used in GraphRAG?", "top_k": 3}
        )
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2))

asyncio.run(main())
