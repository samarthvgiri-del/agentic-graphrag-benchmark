"""
Phase 6A Tests: Benchmark Store and API.
"""

import pytest
import json
import os
import tempfile
from app.db.benchmark_store import BenchmarkStore


@pytest.fixture
def store():
    """Create a temporary benchmark store for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = BenchmarkStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


def test_create_run(store):
    config = {
        "model": "qwen3:4b",
        "embedding_model": "all-MiniLM-L6-v2",
        "qdrant_collection": "documents",
        "tigergraph_graph": "test_graph",
        "question_count": 5,
    }
    run_id = store.create_run(config)
    assert run_id.startswith("run-")

    run = store.get_run(run_id)
    assert run is not None
    assert run["model"] == "qwen3:4b"
    assert run["status"] == "running"


def test_save_and_retrieve_result(store):
    run_id = store.create_run({"model": "test", "question_count": 1})

    result = {
        "run_id": run_id,
        "question_id": "pub-001",
        "question": "What is 2+2?",
        "question_type": "lookup",
        "pipeline": "rag",
        "answer": "4",
        "gold_answer": ["4"],
        "gold_doc_ids": ["doc1"],
        "citations": [{"chunk_id": "c1", "source": "s1"}],
        "retrieved_chunks": [{"chunk_id": "c1", "text": "2+2=4"}],
        "latency_ms": 100,
        "status": "success",
    }
    result_id = store.save_result(result)
    assert result_id.startswith("res-")

    retrieved = store.get_result(result_id)
    assert retrieved is not None
    assert retrieved["question_id"] == "pub-001"
    assert retrieved["pipeline"] == "rag"
    assert retrieved["answer"] == "4"


def test_save_and_retrieve_trace(store):
    run_id = store.create_run({"model": "test", "question_count": 1})
    result_id = store.save_result({
        "run_id": run_id,
        "question_id": "pub-001",
        "question": "Test?",
        "pipeline": "agentic",
        "status": "success",
    })

    trace_data = {
        "step_count": 3,
        "tools_called": ["vector_search", "entity_linking"],
        "strategy_changes": [{"initial_strategy": "initial", "new_strategy": "vector_search", "reason": "test"}],
        "evidence_count": 2,
        "citation_count": 1,
        "stop_reason": "Evidence sufficient",
        "complete_trace": [{"step": 1, "action": "vector_search"}],
        "tokens_per_operation": [],
        "latency_per_operation": [{"step": 1, "action": "vector_search", "latency_ms": 50}],
    }
    trace_id = store.save_trace(result_id, trace_data)
    assert trace_id.startswith("trc-")

    trace = store.get_trace_for_result(result_id)
    assert trace is not None
    assert trace["step_count"] == 3
    assert "vector_search" in trace["tools_called"]


def test_error_isolation(store):
    """Verify that a failed result does not prevent subsequent saves."""
    run_id = store.create_run({"model": "test", "question_count": 3})

    # Success
    r1 = store.save_result({
        "run_id": run_id,
        "question_id": "pub-001",
        "question": "Q1?",
        "pipeline": "rag",
        "answer": "A1",
        "status": "success",
        "latency_ms": 100,
    })

    # Error/timeout
    r2 = store.save_result({
        "run_id": run_id,
        "question_id": "pub-002",
        "question": "Q2?",
        "pipeline": "rag",
        "status": "timeout",
        "error": "Timed out after 45s",
    })

    # Success after error
    r3 = store.save_result({
        "run_id": run_id,
        "question_id": "pub-003",
        "question": "Q3?",
        "pipeline": "rag",
        "answer": "A3",
        "status": "success",
        "latency_ms": 200,
    })

    results = store.get_results_for_run(run_id)
    assert len(results) == 3
    statuses = [r["status"] for r in results]
    assert "success" in statuses
    assert "timeout" in statuses


def test_run_summary(store):
    run_id = store.create_run({"model": "test", "question_count": 2})

    for qid in ["pub-001", "pub-002"]:
        for pipe in ["rag", "graphrag"]:
            store.save_result({
                "run_id": run_id,
                "question_id": qid,
                "question": f"Q {qid}?",
                "pipeline": pipe,
                "answer": "A",
                "status": "success",
                "latency_ms": 100,
                "total_tokens": 50,
            })

    summary = store.get_run_summary(run_id)
    assert "rag" in summary
    assert "graphrag" in summary
    assert summary["rag"]["total"] == 2
    assert summary["rag"]["successes"] == 2


def test_evaluation_storage(store):
    """Test that evaluation results are stored separately from pipeline results."""
    run_id = store.create_run({"model": "test", "question_count": 1})
    result_id = store.save_result({
        "run_id": run_id,
        "question_id": "pub-001",
        "question": "Test?",
        "pipeline": "rag",
        "answer": "4",
        "status": "success",
    })

    eval_id = store.save_evaluation({
        "result_id": result_id,
        "evaluator_version": "v1.0",
        "evaluator_model": "qwen3:4b",
        "evaluation_method": "deterministic_exact_match",
        "correctness": 1.0,
        "completeness": 1.0,
        "groundedness": 0.8,
    })
    assert eval_id.startswith("evl-")

    evals = store.get_evaluations_for_result(result_id)
    assert len(evals) == 1
    assert evals[0]["correctness"] == 1.0
    assert evals[0]["evaluation_method"] == "deterministic_exact_match"


def test_pipeline_filter(store):
    run_id = store.create_run({"model": "test", "question_count": 1})

    store.save_result({
        "run_id": run_id,
        "question_id": "pub-001",
        "question": "Q?",
        "pipeline": "rag",
        "status": "success",
    })
    store.save_result({
        "run_id": run_id,
        "question_id": "pub-001",
        "question": "Q?",
        "pipeline": "graphrag",
        "status": "success",
    })

    rag_only = store.get_results_for_run(run_id, pipeline="rag")
    assert len(rag_only) == 1
    assert rag_only[0]["pipeline"] == "rag"


def test_no_eval_hidden_access():
    """Verify that the benchmark runner does not reference eval_hidden."""
    runner_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "run_benchmark.py"
    )
    with open(runner_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The runner should have a safety assertion but should NOT
    # actually open or load eval_hidden
    assert "eval_hidden" in content  # safety check reference exists
    assert 'load_questions(EVAL_HIDDEN_PATH' not in content  # never called


def test_list_runs(store):
    store.create_run({"model": "m1", "question_count": 1})
    store.create_run({"model": "m2", "question_count": 2})
    runs = store.list_runs()
    assert len(runs) == 2


def test_finish_run(store):
    run_id = store.create_run({"model": "test", "question_count": 1})
    store.finish_run(run_id, "completed")
    run = store.get_run(run_id)
    assert run["status"] == "completed"
