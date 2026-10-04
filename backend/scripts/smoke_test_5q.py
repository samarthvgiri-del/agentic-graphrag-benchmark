import asyncio
import json
import sys
import time

from app.api.routes_rag import RAGQueryRequest, rag_query, graphrag_query
from app.api.routes_agentic import agentic_query

async def main():
    questions_file = "data/questions/eval_public.jsonl"

    questions = []
    types_needed = {'lookup', 'multi-hop', 'aggregation', 'temporal', 'superlative'}
    types_found = set()

    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            q = json.loads(line)
            qtype = q.get("qtype")
            if qtype in types_needed and qtype not in types_found:
                types_found.add(qtype)
                questions.append(q)
            if len(questions) >= 5:
                break

    results = []

    for i, q_data in enumerate(questions):
        q_id = q_data.get("id", f"Q{i+1}")
        q_text = q_data["question"]
        print(f"\n========================================================", flush=True)
        print(f"QUESTION {i+1} [{q_id}]: {q_text}", flush=True)

        request = RAGQueryRequest(question=q_text)

        q_result = {
            "id": q_id,
            "rag": {},
            "graphrag": {},
            "agentic": {}
        }

        # 1. RAG
        print("--- Traditional RAG ---", flush=True)
        start = time.time()
        try:
            rag_resp = await rag_query(request)
            q_result["rag"] = {
                "steps": 1,
                "latency_ms": rag_resp.latency_ms,
                "citations": len(rag_resp.citations)
            }
            print(f"RAG Done. Latency: {rag_resp.latency_ms}ms", flush=True)
        except Exception as e:
            print(f"RAG Error: {e}", flush=True)

        # 2. GraphRAG
        print("--- GraphRAG ---", flush=True)
        try:
            grag_resp = await graphrag_query(request)
            q_result["graphrag"] = {
                "steps": 1,
                "latency_ms": grag_resp.latency_ms,
                "citations": len(grag_resp.citations)
            }
            print(f"GraphRAG Done. Latency: {grag_resp.latency_ms}ms", flush=True)
        except Exception as e:
            print(f"GraphRAG Error: {e}", flush=True)

        # 3. Agentic
        print("--- Agentic GraphRAG ---", flush=True)
        try:
            agent_resp = await agentic_query(request)
            tools_used = [s.action for s in agent_resp.trace.steps]
            strat_changes = len(agent_resp.trace.strategy_changes)
            q_result["agentic"] = {
                "steps": agent_resp.total_steps,
                "latency_ms": agent_resp.latency_ms,
                "citations": len(agent_resp.citations),
                "tools": tools_used,
                "strat_changes": strat_changes,
                "stop_reason": agent_resp.stop_reason
            }
            print(f"Agentic Done. Latency: {agent_resp.latency_ms}ms", flush=True)
            print(f"Steps: {agent_resp.total_steps}, Reason: {agent_resp.stop_reason}", flush=True)
            for sc in agent_resp.trace.strategy_changes:
                print(f"  Strategy: {sc.initial_strategy} -> {sc.new_strategy} ({sc.reason})")
        except Exception as e:
            print(f"Agentic Error: {e}", flush=True)

        results.append(q_result)

        with open("smoke_report.json", "w", encoding="utf-8") as rf:
            json.dump(results, rf, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
