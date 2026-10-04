import os
import sys
import time
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# We need the user to export OPENAI_API_KEY or have it in .env
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY is not set. Please set it in .env and run again.")
    sys.exit(1)

# Must import settings after setting os.environ
from app.config import settings

# Force overrides on the settings object directly to avoid import cache issues
settings.ORCHESTRATOR_PROVIDER = "openai"
settings.EVALUATOR_PROVIDER = "openai"
settings.LLM_PROVIDER = "ollama"

from app.api.routes_agentic import agentic_query
from app.models.schemas import RAGQueryRequest
from app.llm.factory import get_orchestrator_provider, get_evaluator_provider

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
    # Instantiate providers to read their tokens later
    orchestrator_client = get_orchestrator_provider()
    evaluator_client = get_evaluator_provider()

    print(f"Using Orchestrator: {settings.ORCHESTRATOR_PROVIDER} ({settings.ORCHESTRATOR_MODEL or settings.OPENAI_MODEL})")
    print(f"Using Evaluator: {settings.EVALUATOR_PROVIDER} ({settings.EVALUATOR_MODEL or settings.OPENAI_MODEL})")
    print(f"Using Final Answer: {settings.LLM_PROVIDER} ({settings.OLLAMA_MODEL})")
    print("-" * 50)

    metrics = []

    # Let's intercept the providers for the query
    # But agentic_query() gets its own instances via get_*_provider().
    # To read tokens, we can rely on the fact that OpenAIProvider is stateful for tokens if we pass it, but `agentic_query` calls `get_orchestrator_provider()` internally.
    # We should patch `get_orchestrator_provider` and `get_evaluator_provider` in factory during this script.

    import app.api.routes_agentic
    app.api.routes_agentic.get_orchestrator_provider = lambda: orchestrator_client
    app.api.routes_agentic.get_evaluator_provider = lambda: evaluator_client

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        req = RAGQueryRequest(question=q)

        start = time.time()
        # To make sure we capture tokens, we record before and after
        orch_calls_before = orchestrator_client.api_calls
        eval_calls_before = getattr(evaluator_client, 'api_calls', 0)
        in_tok_before = getattr(orchestrator_client, 'input_tokens', 0) + getattr(evaluator_client, 'input_tokens', 0)
        out_tok_before = getattr(orchestrator_client, 'output_tokens', 0) + getattr(evaluator_client, 'output_tokens', 0)

        try:
            resp = await agentic_query(req)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            resp = None

        latency = time.time() - start

        orch_calls = orchestrator_client.api_calls - orch_calls_before
        eval_calls = getattr(evaluator_client, 'api_calls', 0) - eval_calls_before
        in_tok = (getattr(orchestrator_client, 'input_tokens', 0) + getattr(evaluator_client, 'input_tokens', 0)) - in_tok_before
        out_tok = (getattr(orchestrator_client, 'output_tokens', 0) + getattr(evaluator_client, 'output_tokens', 0)) - out_tok_before

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

        metrics.append({
            "question": q,
            "success": success,
            "fallback": fallback,
            "steps": len(tools) if resp else 0,
            "latency": latency,
            "tools": tools,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "orch_calls": orch_calls,
            "eval_calls": eval_calls
        })

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    successes = sum(1 for m in metrics if m["success"])
    fallbacks = sum(1 for m in metrics if m["fallback"])
    genuine = successes - fallbacks
    avg_latency = sum(m["latency"] for m in metrics) / len(metrics)
    avg_steps = sum(m["steps"] for m in metrics) / len(metrics)

    total_in = sum(m["in_tokens"] for m in metrics)
    total_out = sum(m["out_tokens"] for m in metrics)
    cost = (total_in * 0.150 / 1000000) + (total_out * 0.600 / 1000000) # Assuming gpt-4o-mini

    print(f"Structured-output success: {successes}/5")
    print(f"Genuine decision rate: {genuine}/5")
    print(f"Fallback rate: {fallbacks}/5")
    print(f"Avg total Agentic latency: {avg_latency:.2f}s")
    print(f"Avg Agent steps: {avg_steps:.2f}")
    print(f"Total Input Tokens: {total_in}")
    print(f"Total Output Tokens: {total_out}")
    print(f"Estimated Cost: ${cost:.6f}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_test())
