import asyncio
import json
import logging
import sys

from app.config import settings
from app.graph.tigergraph_client import TigerGraphClient
from app.graph.graphrag_pipeline import run_graphrag

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

async def trace_questions():
    questions_to_trace = ["pub-005", "pub-013", "pub-019"]
    questions = []

    with open("data/questions/eval_public.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if data["qid"] in questions_to_trace:
                questions.append(data)

    client = TigerGraphClient()

    for q in questions:
        print(f"\n{'='*60}")
        print(f"TRACING QUESTION: {q['qid']} - {q['question']}")
        print(f"{'='*60}")

        result = await run_graphrag(q['question'], client)
        print(f"\nFINAL ANSWER:\n{result['answer']}")
        print(f"CITATIONS:\n{result['citations']}")

if __name__ == "__main__":
    asyncio.run(trace_questions())
