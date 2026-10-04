"""
Phase 6B Evaluation Runner.

Runs evaluation on the results of a specific benchmark run.
"""

import asyncio
import sys
import os
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.benchmark_store import BenchmarkStore
from app.eval.evaluator import Evaluator
from app.llm.ollama_client import ollama_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def run_evaluation(run_id: str):
    store = BenchmarkStore()
    run = store.get_run(run_id)
    if not run:
        logger.error(f"Run {run_id} not found.")
        sys.exit(1)

    results = store.get_results_for_run(run_id)
    logger.info(f"Evaluating {len(results)} results for run {run_id}...")

    evaluator = Evaluator(ollama_client)

    for r in tqdm(results):
        # Check if already evaluated
        existing = store.get_evaluations_for_result(r["result_id"])
        if existing:
            continue

        eval_data = await evaluator.evaluate(r)
        store.save_evaluation(eval_data)

    logger.info("Evaluation complete.")
    store.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.run_id))
