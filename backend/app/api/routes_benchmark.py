"""
API routes for benchmark results (Phase 6A).

Provides read-only access to benchmark runs, results, traces, and summaries.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.db.benchmark_store import BenchmarkStore

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


def _get_store() -> BenchmarkStore:
    return BenchmarkStore()


@router.get("/runs")
def list_runs():
    """List all benchmark runs."""
    store = _get_store()
    try:
        return store.list_runs()
    finally:
        store.close()


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """Get a specific benchmark run."""
    store = _get_store()
    try:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run
    finally:
        store.close()


@router.get("/runs/{run_id}/summary")
def get_run_summary(run_id: str):
    """Get aggregate statistics for a run, grouped by pipeline."""
    store = _get_store()
    try:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return store.get_run_summary(run_id)
    finally:
        store.close()


@router.get("/runs/{run_id}/results")
def get_run_results(
    run_id: str,
    pipeline: Optional[str] = Query(None, description="Filter by pipeline"),
):
    """List all results for a benchmark run."""
    store = _get_store()
    try:
        results = store.get_results_for_run(run_id, pipeline=pipeline)
        return results
    finally:
        store.close()


@router.get("/results/{result_id}")
def get_result(result_id: str):
    """Get a specific result."""
    store = _get_store()
    try:
        result = store.get_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result
    finally:
        store.close()


@router.get("/results/{result_id}/trace")
def get_result_trace(result_id: str):
    """Get the agentic trace for a specific result."""
    store = _get_store()
    try:
        trace = store.get_trace_for_result(result_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace
    finally:
        store.close()


@router.get("/results/{result_id}/evaluations")
def get_result_evaluations(result_id: str):
    """Get all evaluations for a specific result."""
    store = _get_store()
    try:
        evals = store.get_evaluations_for_result(result_id)
        return evals
    finally:
        store.close()
