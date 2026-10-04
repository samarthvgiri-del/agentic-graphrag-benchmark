import time
from typing import Optional
from app.agents.state import AgentState
from app.models.schemas import EvidenceEvalResult, OrchestratorAction

class StoppingPolicy:
    def __init__(self, max_steps: int = 5, max_runtime_ms: int = 60000, max_tool_repeats: int = 2):
        self.max_steps = max_steps
        self.max_runtime_ms = max_runtime_ms
        self.max_tool_repeats = max_tool_repeats

    def should_stop(self, state: AgentState, eval_result: EvidenceEvalResult) -> bool:
        if eval_result.sufficient:
            state.set_finished("Evidence is sufficient to answer.")
            return True

        if len(state.steps) >= self.max_steps:
            state.set_finished("Maximum steps reached.")
            return True

        runtime_ms = int((time.time() - state.start_time) * 1000)
        if runtime_ms >= self.max_runtime_ms:
            state.set_finished("Maximum runtime budget reached.")
            return True

        if eval_result.recommended_next_action == "finish":
            state.set_finished("Evidence evaluator recommended stopping.")
            return True

        return False

    def should_block_action(self, state: AgentState, action: OrchestratorAction) -> Optional[str]:
        """Check if a specific orchestrator action should be blocked.

        Returns a reason string if blocked, None if allowed.
        """
        # Block if this exact action+target already failed
        action_key = f"{action.action}:{action.target}"
        if action_key in state.failed_actions:
            return f"Action '{action.action}' on '{action.target}' already failed"

        # Block if this tool type has been used too many times
        count = state.tool_calls_count.get(action.action, 0)
        if count >= self.max_tool_repeats:
            return f"Tool '{action.action}' already used {count} times (max={self.max_tool_repeats})"

        return None
