"""
Regression test for Agentic Adaptivity (Phase 6 fixes).
"""

import pytest
from app.agents.stopping import StoppingPolicy
from app.agents.state import AgentState
from app.models.schemas import OrchestratorAction, EvidenceEvalResult

def test_stopping_policy_prevents_failed_action_repeats():
    policy = StoppingPolicy(max_steps=5, max_tool_repeats=2)
    state = AgentState("Test question")

    # Simulate a failed action
    state.failed_actions.add("entity_linking:Test entity")

    # Try the exact same action again
    action1 = OrchestratorAction(
        action="entity_linking",
        target="Test entity",
        reason="Trying again",
        expected_information="Info"
    )

    # Should be blocked
    block_reason = policy.should_block_action(state, action1)
    assert block_reason is not None
    assert "already failed" in block_reason

    # Try a different target with the same tool
    action2 = OrchestratorAction(
        action="entity_linking",
        target="Different entity",
        reason="New target",
        expected_information="Info"
    )

    # Should NOT be blocked (unless max tool repeats hit)
    block_reason2 = policy.should_block_action(state, action2)
    assert block_reason2 is None

def test_stopping_policy_tool_exhaustion():
    policy = StoppingPolicy(max_steps=5, max_tool_repeats=2)
    state = AgentState("Test question")

    # Simulate tool used 2 times
    state.tool_calls_count["document_retrieval"] = 2

    action = OrchestratorAction(
        action="document_retrieval",
        target="Some search",
        reason="Searching",
        expected_information="Info"
    )

    block_reason = policy.should_block_action(state, action)
    assert block_reason is not None
    assert "already used 2 times" in block_reason
