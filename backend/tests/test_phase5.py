import pytest
import time
from app.models.schemas import OrchestratorAction, EvidenceEvalResult, NormalizedToolResult
from app.agents.state import AgentState
from app.agents.stopping import StoppingPolicy
from app.agents.context import ContextManager

def test_stopping_policy_sufficient_evidence():
    policy = StoppingPolicy(max_steps=5, max_tool_repeats=2)
    state = AgentState(question="Test")
    eval_result = EvidenceEvalResult(
        sufficient=True,
        confidence=0.9,
        missing_information=[],
        contradiction_detected=False,
        recommended_next_action="finish"
    )
    should_stop = policy.should_stop(state, eval_result)
    assert should_stop is True
    assert state.is_finished is True
    assert "sufficient" in state.stop_reason.lower()

def test_stopping_policy_max_steps():
    policy = StoppingPolicy(max_steps=2, max_tool_repeats=5)
    state = AgentState(question="Test")
    state.steps = [{"step": 1}, {"step": 2}]

    eval_result = EvidenceEvalResult(
        sufficient=False,
        confidence=0.1,
        missing_information=["More info needed"],
        contradiction_detected=False,
        recommended_next_action="vector_search"
    )
    should_stop = policy.should_stop(state, eval_result)
    assert should_stop is True
    assert "Maximum steps" in state.stop_reason

def test_stopping_policy_tool_repeats():
    policy = StoppingPolicy(max_steps=5, max_tool_repeats=2)
    state = AgentState("Test question")
    state.tool_calls_count = {"vector_search": 2}

    # Tool repeats are now checked via should_block_action
    action = OrchestratorAction(
        action="vector_search",
        target="target",
        reason="reason",
        expected_information="info"
    )
    block_reason = policy.should_block_action(state, action)
    assert block_reason is not None
    assert "already used 2 times" in block_reason

def test_context_budget():
    cm = ContextManager(max_items=2, max_chars=100, max_tokens=25)
    state = AgentState(question="Test")

    # Add tool results to state
    res1 = NormalizedToolResult(
        tool="test_tool",
        success=True,
        results=[],
        evidence=["This is a very long piece of evidence that exceeds the character limit easily when combined."],
        source_ids=["src1"],
        latency_ms=10
    )
    res2 = NormalizedToolResult(
        tool="test_tool",
        success=True,
        results=[],
        evidence=["Another piece of evidence."],
        source_ids=["src2"],
        latency_ms=10
    )
    action = OrchestratorAction(action="test_tool", target="test", reason="test", expected_information="test")

    state.add_tool_result(action, res1)
    state.add_tool_result(action, res2)

    context_str = cm.build_context_string(state)
    assert len(context_str) <= 125
    assert "[TRUNCATED]" in context_str

def test_agent_state_deduplication():
    state = AgentState(question="Test")
    res1 = NormalizedToolResult(
        tool="vector_search",
        success=True,
        results=[],
        evidence=["Exact same fact."],
        source_ids=["src1"],
        latency_ms=10
    )
    res2 = NormalizedToolResult(
        tool="graph_traversal",
        success=True,
        results=[],
        evidence=["Exact same fact."],
        source_ids=["src1"], # same ID
        latency_ms=10
    )

    action1 = OrchestratorAction(action="vector_search", target="test", reason="test", expected_information="test")
    action2 = OrchestratorAction(action="graph_traversal", target="test", reason="test", expected_information="test")

    state.add_tool_result(action1, res1)
    state.add_tool_result(action2, res2)

    # Should only have 1 piece of evidence despite 2 tools returning it
    assert len(state.candidate_evidence) == 1
