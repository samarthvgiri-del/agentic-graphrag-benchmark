"""
Standalone Benchmark Runner (Independent Execution).

Requirements:
1. Run from terminal as standalone Python process.
2. Batched processing.
3. Persist immediately (via BenchmarkStore).
4. Auto-resume and skip completed.
5. Retry failed questions separately.
6. Print detailed progress [X/Y] PIPELINE completed.
7. Validation mode (first 10 questions).
"""

import asyncio
import json
import sys
import os
import argparse
import logging
from datetime import datetime, timezone

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.benchmark_store import BenchmarkStore
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EVAL_COMBINED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "questions", "eval_combined.jsonl"
)
EVAL_PUBLIC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "questions", "eval_public.jsonl"
)


def load_questions(path: str) -> list:
    questions = []
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return questions

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def build_run_config(q_count: int) -> dict:
    return {
        "model": settings.OLLAMA_MODEL,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "qdrant_collection": "documents",
        "tigergraph_graph": settings.TIGERGRAPH_GRAPHNAME,
        "agent_limits": {
            "max_steps": 5,
            "max_tool_repeats": 2,
            "max_context_chars": 12000,
            "max_context_tokens": 3000,
            "ollama_timeout_ms": settings.OLLAMA_REQUEST_TIMEOUT_MS,
            "agent_budget_ms": settings.AGENT_RUNTIME_BUDGET_MS,
        },
        "inference_settings": {
            "rag_temperature": 0.3,
            "rag_top_k": 3,
            "graphrag_temperature": 0.2,
            "agentic_temperature": 0.2,
        },
        "question_count": q_count
    }


async def run_rag(question: str) -> dict:
    from app.api.routes_rag import rag_query
    from app.models.schemas import RAGQueryRequest
    request = RAGQueryRequest(question=question)
    resp = await rag_query(request)
    return {
        "answer": resp.answer,
        "citations": [c.dict() if hasattr(c, "dict") else c for c in resp.citations],
        "retrieved_chunks": [rc.dict() if hasattr(rc, "dict") else rc for rc in resp.retrieved_chunks],
        "latency_ms": int(resp.latency_ms),
        "retrieved_chunk_count": len(resp.retrieved_chunks),
        "unique_source_count": len(set(c.source for c in resp.retrieved_chunks)),
    }


async def run_graphrag(question: str) -> dict:
    from app.api.routes_rag import graphrag_query
    from app.models.schemas import RAGQueryRequest
    request = RAGQueryRequest(question=question)
    resp = await graphrag_query(request)
    return {
        "answer": resp.answer,
        "citations": resp.citations,
        "retrieved_chunks": resp.retrieved_chunks,
        "latency_ms": int(resp.latency_ms),
        "extracted_phrases": getattr(resp, "extracted_phrases", []),
        "linked_entities": getattr(resp, "linked_entities", []),
        "graph_nodes_returned": getattr(resp, "graph_nodes_returned", 0),
        "retrieved_chunk_count": len(resp.retrieved_chunks),
        "unique_source_count": len(set(c.get("source", "unknown") if isinstance(c, dict) else getattr(c, "source", "unknown") for c in resp.retrieved_chunks))
    }


async def run_agentic(question: str) -> dict:
    from app.api.routes_agentic import agentic_query
    from app.models.schemas import RAGQueryRequest
    request = RAGQueryRequest(question=question)
    resp = await agentic_query(request)

    trace = resp.trace
    steps_data = [s.dict() if hasattr(s, "dict") else s for s in trace.steps]
    return {
        "answer": resp.answer,
        "citations": resp.citations,
        "retrieved_chunks": resp.retrieved_chunks,
        "latency_ms": int(resp.latency_ms),
        "orchestrator_fallback_used": any(s.get("fallback_used", False) if isinstance(s, dict) else getattr(s, "fallback_used", False) for s in steps_data),
        "retrieved_chunk_count": len(resp.retrieved_chunks),
        "unique_source_count": len(set(c.get("source", "unknown") if isinstance(c, dict) else getattr(c, "source", "unknown") for c in resp.retrieved_chunks)),
        "trace": {
            "step_count": trace.total_steps,
            "tools_called": trace.tools_called,
            "complete_trace": steps_data,
        },
    }

PIPELINE_RUNNERS = {
    "rag": run_rag,
    "graphrag": run_graphrag,
    "agentic": run_agentic,
}


async def run_single(store: BenchmarkStore, run_id: str, q_data: dict, pipeline: str, evaluator) -> dict:
    qid = q_data["qid"]
    question = q_data["question"]
    gold_answer = q_data.get("answer", [])

    result_record = {
        "run_id": run_id,
        "question_id": qid,
        "question": question,
        "pipeline": pipeline,
        "gold_answer": gold_answer,
        "status": "error",
    }

    try:
        runner = PIPELINE_RUNNERS[pipeline]
        pipeline_result = await runner(question)

        result_record.update({
            "answer": pipeline_result.get("answer"),
            "citations": pipeline_result.get("citations", []),
            "retrieved_chunks": pipeline_result.get("retrieved_chunks", []),
            "latency_ms": pipeline_result.get("latency_ms", 0),
            "status": "success",
            "extracted_phrases": pipeline_result.get("extracted_phrases", []),
            "linked_entities": pipeline_result.get("linked_entities", []),
            "graph_nodes_returned": pipeline_result.get("graph_nodes_returned", 0),
            "retrieved_chunk_count": pipeline_result.get("retrieved_chunk_count", 0),
            "unique_source_count": pipeline_result.get("unique_source_count", 0),
            "orchestrator_fallback_used": pipeline_result.get("orchestrator_fallback_used", False)
        })

        result_id = store.save_result(result_record)
        if pipeline == "agentic" and "trace" in pipeline_result:
            store.save_trace(result_id, pipeline_result["trace"])

        # Evaluator
        result_record["result_id"] = result_id
        eval_res = await evaluator.evaluate(result_record)
        store.save_evaluation(eval_res)

        return {"status": "success", "latency": pipeline_result.get("latency_ms", 0)}

    except Exception as exc:
        result_record["error"] = str(exc)
        result_record["status"] = "error"
        result_id = store.save_result(result_record)

        result_record["result_id"] = result_id
        try:
            eval_res = await evaluator.evaluate(result_record)
            store.save_evaluation(eval_res)
        except:
            pass

        return {"status": "error", "error": str(exc), "latency": 0}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", action="store_true", help="Run 10 questions only")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process")
    parser.add_argument("--dataset", type=str, default="combined", choices=["combined", "public"], help="Dataset to run against")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for processing")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously failed questions")
    parser.add_argument("--run-id", type=str, default="run-default", help="Persistent Run ID to resume from")
    args = parser.parse_args()

    dataset_path = EVAL_PUBLIC_PATH if args.dataset == "public" else EVAL_COMBINED_PATH
    assert "eval_hidden" not in dataset_path, "eval_hidden should not be loaded directly"
    questions = load_questions(dataset_path)
    if not questions:
        return

    if args.validation:
        questions = questions[:10]
        logger.info("=== VALIDATION MODE (10 Questions) ===")
    elif args.limit:
        questions = questions[:args.limit]
        logger.info(f"=== LIMITED MODE ({args.limit} Questions) ===")

    total = len(questions)
    store = BenchmarkStore()

    # Ensure run_id exists
    run_exists = store.conn.execute("SELECT 1 FROM benchmark_runs WHERE run_id = ?", (args.run_id,)).fetchone()
    if not run_exists:
        config = build_run_config(total)
        store.conn.execute(
            """INSERT INTO benchmark_runs
               (run_id, timestamp, model, embedding_model, qdrant_collection,
                tigergraph_graph, agent_limits, inference_settings, question_count, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                args.run_id,
                datetime.now(timezone.utc).isoformat(),
                config.get("model"),
                config.get("embedding_model"),
                config.get("qdrant_collection"),
                config.get("tigergraph_graph"),
                json.dumps(config.get("agent_limits", {})),
                json.dumps(config.get("inference_settings", {})),
                config.get("question_count", 0),
                "running",
            ),
        )
        store.conn.commit()

    # Get completed questions for this run to skip them
    existing_results = store.conn.execute(
        "SELECT question_id, pipeline, status FROM benchmark_results WHERE run_id = ?",
        (args.run_id,)
    ).fetchall()

    completed = set()
    failed = set()
    for row in existing_results:
        key = f"{row['question_id']}-{row['pipeline']}"
        if row['status'] == 'success':
            completed.add(key)
        else:
            failed.add(key)

    from app.eval.evaluator import Evaluator
    from app.llm.factory import get_evaluator_provider
    evaluator = Evaluator(get_evaluator_provider())

    logger.info(f"Starting benchmark {args.run_id} | Total questions: {total}")

    # Process in batches
    for batch_start in range(0, total, args.batch_size):
        batch = questions[batch_start:batch_start + args.batch_size]
        logger.info(f"\n--- Starting Batch {batch_start//args.batch_size + 1} ---")

        for q_idx, q_data in enumerate(batch):
            abs_idx = batch_start + q_idx + 1
            qid = q_data["qid"]

            for pipeline in ["rag", "graphrag", "agentic"]:
                key = f"{qid}-{pipeline}"

                # Check skip logic
                if key in completed:
                    logger.info(f"[{abs_idx}/{total}] {pipeline.upper()} skipped (already completed)")
                    continue
                if key in failed and not args.retry_failed:
                    logger.info(f"[{abs_idx}/{total}] {pipeline.upper()} skipped (failed previously, use --retry-failed)")
                    continue

                # Run the pipeline
                res = await run_single(store, args.run_id, q_data, pipeline, evaluator)
                if res["status"] == "success":
                    logger.info(f"[{abs_idx}/{total}] {pipeline.upper()} completed in {res['latency']}ms")
                else:
                    logger.info(f"[{abs_idx}/{total}] {pipeline.upper()} ERROR: {res.get('error')}")

    # Summary
    logger.info("\n=== RUN SUMMARY ===")
    summary = store.get_run_summary(args.run_id)
    for pipe, stats in summary.items():
        corr = stats.get('avg_correctness')
        comp = stats.get('avg_completeness')
        grou = stats.get('avg_groundedness')
        corr_str = f"{corr:.2f}" if corr is not None else "N/A"
        comp_str = f"{comp:.2f}" if comp is not None else "N/A"
        grou_str = f"{grou:.2f}" if grou is not None else "N/A"

        logger.info(f"\n{pipe.upper()}")
        logger.info(f"Execution: {stats['successes']}/{stats['total']}")
        logger.info(f"Evaluated: {stats.get('eval_successes', 0)}/{stats.get('eval_total', 0)}")
        logger.info(f"Evaluator errors: {stats.get('eval_errors', 0)}")
        logger.info(f"Correctness: {corr_str}")
        logger.info(f"Completeness: {comp_str}")
        logger.info(f"Groundedness: {grou_str}")
        logger.info(f"Latency: {stats.get('avg_latency_ms', 0):.0f} ms")
        if pipe == "agentic":
            logger.info(f"Agent Steps: {stats.get('avg_agent_steps', 0):.1f}")


    store.close()

if __name__ == "__main__":
    asyncio.run(main())
