"""
Tests for Phase 6B Evaluator.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.eval.evaluator import Evaluator

@pytest.fixture
def mock_ollama():
    mock = AsyncMock()
    mock.model = "test-model"
    mock.generate = AsyncMock()
    return mock

@pytest.fixture
def evaluator(mock_ollama):
    return Evaluator(mock_ollama)

def test_normalize_text(evaluator):
    assert evaluator.normalize_text("Hello, World!") == "hello world"
    assert evaluator.normalize_text("  123.45  ") == "12345"
    assert evaluator.normalize_text("A-B_C") == "abc"

def test_deterministic_correctness_numeric(evaluator):
    # Single number match
    assert evaluator.check_deterministic_correctness("The answer is 5.", ["5"]) == 1.0

    # Multiple numbers in generation - uncertain
    assert evaluator.check_deterministic_correctness("It could be 5 or 6.", ["5"]) is None

    # Number not present
    assert evaluator.check_deterministic_correctness("The answer is 6.", ["5"]) == 0.0

def test_deterministic_correctness_entity(evaluator):
    # Entity exact substring match
    assert evaluator.check_deterministic_correctness("The winner was Chen Ding.", ["Chen Ding"]) == 1.0

    # Normalization helps
    assert evaluator.check_deterministic_correctness("Winner: CHEN DING!", ["Chen Ding"]) == 1.0

    # Entity not found
    assert evaluator.check_deterministic_correctness("The winner was John Doe.", ["Chen Ding"]) is None

@pytest.mark.asyncio
async def test_semantic_correctness(evaluator):
    evaluator.llm_client.generate.return_value = '{"correct": true, "reason": "ok"}'
    score = await evaluator.evaluate_semantic_correctness("Q", "Gen", ["Gold"])
    assert score == 1.0

    evaluator.llm_client.generate.return_value = '{"correct": false, "reason": "bad"}'
    score = await evaluator.evaluate_semantic_correctness("Q", "Gen", ["Gold"])
    assert score == 0.0

def test_completeness(evaluator):
    assert evaluator.evaluate_completeness("A and B", ["A"]) == 1.0
    assert evaluator.evaluate_completeness("A and B", ["A", "B"]) == 1.0
    assert evaluator.evaluate_completeness("Only A", ["A", "B"]) == 0.5
    assert evaluator.evaluate_completeness("Nothing", ["A", "B"]) == 0.0

@pytest.mark.asyncio
async def test_evaluate_insufficient_evidence(evaluator):
    result = {
        "result_id": "res-1",
        "answer": "Insufficient evidence to answer the question.",
        "status": "success",
        "gold_answer": ["5"],
        "retrieved_chunks": []
    }

    eval_data = await evaluator.evaluate(result)
    assert eval_data["evaluation_method"] == "deterministic"
    assert eval_data["correctness"] == 0.0
    assert eval_data["completeness"] == 0.0
    assert eval_data["groundedness"] == 0.0

@pytest.mark.asyncio
async def test_evaluate_deterministic_success(evaluator):
    result = {
        "result_id": "res-2",
        "question": "How many?",
        "answer": "5",
        "status": "success",
        "gold_answer": ["5"],
        "retrieved_chunks": [{"text": "There are 5."}]
    }

    evaluator.llm_client.generate.return_value = '{"grounded": true, "reason": "ok"}'

    eval_data = await evaluator.evaluate(result)
    assert eval_data["evaluation_method"] == "deterministic"
    assert eval_data["correctness"] == 1.0
    assert eval_data["completeness"] == 1.0
    assert eval_data["groundedness"] == 1.0
    assert evaluator.llm_client.generate.call_count == 1 # only called for groundedness
