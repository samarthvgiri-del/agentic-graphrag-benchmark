import os
import sys
import time
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Force overrides to use local Ollama for everything
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["ORCHESTRATOR_PROVIDER"] = "ollama"
os.environ["EVALUATOR_PROVIDER"] = "ollama"

# Must import settings after setting os.environ
from app.config import settings

settings.LLM_PROVIDER = "ollama"
settings.ORCHESTRATOR_PROVIDER = "ollama"
settings.EVALUATOR_PROVIDER = "ollama"

from app.api.routes_agentic import agentic_query
from app.models.schemas import RAGQueryRequest

questions = [
    # 1. lookup
    "Who directed the movie Inception?",
    # 2. multi-hop
    "What is the capital of the country where the inventor of the telephone was born?",
    # 3. aggregation
    "How many medals did the US win in the 2008 Olympics?",
    # 4. temporal
    "Who was the president of the US during the Moon landing?",
    # 5. superlative
    "What is the tallest mountain in the world?"
]

async def run_test():
    print(f"Using Orchestrator: {settings.ORCHESTRATOR_PROVIDER} ({settings.ORCHESTRATOR_MODEL or settings.OLLAMA_MODEL})")
    print(f"Using Evaluator: {settings.EVALUATOR_PROVIDER} ({settings.EVALUATOR_MODEL or settings.OLLAMA_MODEL})")
    print(f"Using Final Answer: {settings.LLM_PROVIDER} ({settings.OLLAMA_MODEL})")
    print("-" * 50)

    metrics = []

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        req = RAGQueryRequest(question=q)

        start = time.time()
        try:
            resp = await agentic_query(req)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            resp = None

        latency = time.time() - start

        fallback = False
        tools = []
        if resp:
            trace = resp.trace
            steps = [s.dict() if hasattr(s, "dict") else s for s in trace.steps]
            fallback = any(s.get("fallback_used") for s in steps)
            tools = trace.tools_called
            print(f"Tools sequence: {tools}")
            print(f"Fallback: {fallback}")
            print(f"Latency: {latency:.2f}s")
        else:
            print(f"Error: {error}")

        metrics.append({
            "question": q,
            "success": success,
            "fallback": fallback,
            "steps": len(tools) if resp else 0,
            "latency": latency,
            "tools": tools,
        })

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    successes = sum(1 for m in metrics if m["success"])
    fallbacks = sum(1 for m in metrics if m["fallback"])
    genuine = successes - fallbacks
    avg_latency = sum(m["latency"] for m in metrics) / len(metrics) if metrics else 0
    avg_steps = sum(m["steps"] for m in metrics) / len(metrics) if metrics else 0

    print(f"Structured-output success: {successes}/5")
    print(f"Genuine decision rate: {genuine}/5")
    print(f"Fallback rate: {fallbacks}/5")
    print(f"Avg total Agentic latency: {avg_latency:.2f}s")
    print(f"Avg Agent steps: {avg_steps:.2f}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_test())
