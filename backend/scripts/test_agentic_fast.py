import asyncio
import json
import sys
import os

from app.api.routes_rag import RAGQueryRequest
from app.api.routes_agentic import agentic_query

async def main():
    q = "Who won a gold medal in the 2004 Summer Olympics for the United States?"
    print(f"QUESTION: {q}")
    request = RAGQueryRequest(question=q)

    print("--- Agentic GraphRAG ---")
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
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
