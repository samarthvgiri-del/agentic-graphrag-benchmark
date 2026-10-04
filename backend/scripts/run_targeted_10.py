import asyncio
import json
import logging
from typing import List, Dict, Any

from app.rag.pipeline import run_rag_pipeline
from app.rag.vector_store import build_vector_store
from app.graph.graphrag_pipeline import run_graphrag
from app.api.routes_agentic import agentic_query
from app.models.schemas import RAGQueryRequest
from app.graph.tigergraph_client import TigerGraphClient
from app.eval.evaluator import Evaluator
from app.llm.ollama_client import ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_QIDS = {
    "pub-003", "pub-012",  # Lookup
    "pub-004", "pub-022",  # Temporal
    "pub-015", "pub-031",  # Aggregation
    "pub-008", "pub-018",  # Multi-hop
    "pub-006", "pub-037"   # Superlative
}

async def run_benchmark():
    questions = []
    with open("data/questions/eval_public.jsonl", "r") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if data.get("qid") in TARGET_QIDS:
                questions.append(data)

    if len(questions) != 10:
        logger.error(f"Found {len(questions)} questions instead of 10. Check IDs.")
        return

    store = build_vector_store()
    tg_client = TigerGraphClient()
    evaluator = Evaluator(ollama_client)

    results = []

    for q in questions:
        logger.info(f"\n============================")
        logger.info(f"Question {q['qid']}: {q['question']}")
        logger.info(f"Gold Answer: {q['answer']}")

        pipelines = [
            ("rag", lambda q_text: run_rag_pipeline(q_text, store, top_k=5)),
            ("graphrag", lambda q_text: run_graphrag(q_text, tg_client)),
            ("agentic", lambda q_text: agentic_query(RAGQueryRequest(question=q_text, top_k=5)))
        ]

        for p_name, p_func in pipelines:
            logger.info(f"--- Running {p_name} ---")
            try:
                p_res = await p_func(q["question"])

                if isinstance(p_res, dict):
                    status = "success"
                else:
                    p_res = p_res.model_dump()
                    status = "success"

            except Exception as e:
                logger.error(f"Pipeline {p_name} failed: {e}")
                p_res = {"answer": "", "latency_ms": 0, "citations": []}
                status = "error"

            eval_input = {
                "result_id": f"{q['qid']}_{p_name}",
                "question": q["question"],
                "gold_answer": q["answer"],
                "answer": p_res.get("answer", ""),
                "status": status,
                "retrieved_chunks": p_res.get("retrieved_chunks", [])
            }

            eval_res = await evaluator.evaluate(eval_input)

            res_record = {
                "question_id": q["qid"],
                "pipeline": p_name,
                "status": status,
                "answer": p_res.get("answer", ""),
                "latency_ms": p_res.get("latency_ms", 0),
                "citations": p_res.get("citations", []),
                "evaluation": eval_res
            }
            if p_name == "agentic":
                res_record["trace"] = p_res.get("trace", {})

            results.append(res_record)

            # Print brief evaluation
            score_correct = eval_res.get("correctness")
            score_ground = eval_res.get("groundedness")
            logger.info(f"Result: {status}, Correctness: {score_correct}, Groundedness: {score_ground}, Latency: {res_record['latency_ms']}ms")

    with open("results_10_targeted.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Done! Results saved to results_10_targeted.json")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
