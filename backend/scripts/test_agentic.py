import asyncio
import json
import os
from pprint import pprint

from app.api.routes_rag import RAGQueryRequest, rag_query, graphrag_query
from app.api.routes_agentic import agentic_query

async def main():
    questions_file = "data/questions/eval_public (1).jsonl"
    if not os.path.exists(questions_file):
        print(f"Error: Could not find {questions_file}")
        return

    # Read first 3 questions
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            q = json.loads(line)
            questions.append(q["question"])
            if len(questions) >= 3:
                break

    for i, q in enumerate(questions):
        print(f"\n========================================================")
        print(f"QUESTION {i+1}: {q}")
        print(f"========================================================")

        request = RAGQueryRequest(question=q)

        # 1. RAG
        print("\n--- Traditional RAG ---")
        rag_resp = await rag_query(request)
        print(f"Answer: {rag_resp.answer[:200]}...")
        print(f"Latency: {rag_resp.latency_ms} ms, Citations: {len(rag_resp.citations)}")

        # 2. GraphRAG
        print("\n--- GraphRAG ---")
        try:
            grag_resp = await graphrag_query(request)
            print(f"Answer: {grag_resp.answer[:200]}...")
            print(f"Latency: {grag_resp.latency_ms} ms, Citations: {len(grag_resp.citations)}")
        except Exception as e:
            print(f"GraphRAG failed: {e}")

        # 3. Agentic GraphRAG
        print("\n--- Agentic GraphRAG ---")
        try:
            agent_resp = await agentic_query(request)
            print(f"Answer: {agent_resp.answer[:200]}...")
            print(f"Total Steps: {agent_resp.total_steps}")
            print(f"Latency: {agent_resp.latency_ms} ms")
            print(f"Stop Reason: {agent_resp.stop_reason}")
            print("\nTrace Steps:")
            for s in agent_resp.trace.steps:
                print(f"  Step {s.step}: {s.action} (Reason: {s.reason})")
                print(f"    Target: {s.input_summary}")
                print(f"    Result: {s.result_summary}")
                print(f"    Latency: {s.latency_ms}ms")
            print("\nStrategy Changes:")
            for sc in agent_resp.trace.strategy_changes:
                print(f"  {sc.initial_strategy} -> {sc.new_strategy} ({sc.reason})")
        except Exception as e:
            print(f"Agentic GraphRAG failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
