import time
from typing import List, Dict, Any, Optional

from app.models.schemas import (
    AgenticTraceStep, StrategyChange, OrchestratorAction, NormalizedToolResult
)

class AgentState:
    """
    Holds the state of the Agentic GraphRAG orchestrator loop.
    """
    def __init__(self, question: str):
        self.question: str = question

        # Extracted structured data
        self.entities: List[Dict[str, Any]] = []
        self.relationships: List[Dict[str, Any]] = []
        self.retrieved_chunks: List[Dict[str, Any]] = []

        # Candidate evidence for the context budget
        # format: {"id": str, "text": str, "source": str, "type": str}
        self.candidate_evidence: List[Dict[str, Any]] = []
        self.seen_evidence_ids: set = set()

        # State tracking
        self.steps: List[AgenticTraceStep] = []
        self.tools_called: List[str] = []
        self.strategy_changes: List[StrategyChange] = []
        self.tool_calls_count: Dict[str, int] = {}

        self.failed_actions: set = set()
        self.successful_actions: set = set()

        # Current status
        self.current_strategy: str = "initial"
        self.stop_reason: Optional[str] = None
        self.is_finished: bool = False

        self.start_time: float = time.time()
        self.total_latency_ms: int = 0

    def record_step(self, action: OrchestratorAction, latency_ms: int,
                    status: str, input_size: int, output_size: int,
                    candidate_count: int, result_count: int, think_enabled: bool,
                    timeout_val: float):
        from datetime import datetime

        step_obj = AgenticTraceStep(
            step=len(self.steps) + 1,
            action=action.action,
            target=action.target,
            reason=action.reason,
            start_time=datetime.utcnow().isoformat(),
            timeout=timeout_val,
            status=status,
            latency_ms=latency_ms,
            input_size=input_size,
            output_size=output_size,
            candidate_count=candidate_count,
            result_count=result_count,
            think_enabled=think_enabled,
            fallback_used=getattr(action, "fallback_used", False)
        )
        self.steps.append(step_obj)
        self.total_latency_ms += latency_ms

    def record_strategy_change(self, reason: str, new_strategy: str):
        if self.current_strategy != new_strategy:
            sc = StrategyChange(
                initial_strategy=self.current_strategy,
                reason=reason,
                new_strategy=new_strategy
            )
            self.strategy_changes.append(sc)
            self.current_strategy = new_strategy

    def add_tool_result(self, action: OrchestratorAction, result: NormalizedToolResult):
        self.tools_called.append(result.tool)
        self.tool_calls_count[result.tool] = self.tool_calls_count.get(result.tool, 0) + 1

        action_key = f"{action.action}:{action.target}"
        if result.success and result.results:
            self.successful_actions.add(action_key)
        else:
            self.failed_actions.add(action_key)

        for idx, ev in enumerate(result.evidence):
            source_id = result.source_ids[idx] if idx < len(result.source_ids) else f"{result.tool}_{len(self.candidate_evidence)}"
            if source_id not in self.seen_evidence_ids:
                self.candidate_evidence.append({
                    "id": source_id,
                    "text": ev,
                    "source": result.tool,
                    "type": "text"
                })
                self.seen_evidence_ids.add(source_id)

    def set_finished(self, reason: str):
        self.is_finished = True
        self.stop_reason = reason
