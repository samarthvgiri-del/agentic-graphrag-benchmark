import json
import logging
import re
from app.models.schemas import OrchestratorAction, EvidenceEvalResult
from app.agents.state import AgentState
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)

ORCHESTRATOR_PROMPT = """Decide the next action to answer the question.

Question: {question}
History:
{step_history}

Status:
- Sufficient: {eval_sufficient}
- Missing: {eval_missing}
- Collected: {evidence_count} items

Available: {available_tools}

RULES:
1. If Sufficient=True, action MUST be "stop".
2. Pick from Available tools. NEVER pick an exhausted tool.
3. Try different targets if previous failed.

Output valid JSON matching this schema:
{{
  "action": "entity_linking" | "vector_search" | "graph_traversal" | "document_retrieval" | "stop",
  "target": "Query/entity to search",
  "reason": "Brief reason"
}}
"""


def _build_step_history(state: AgentState) -> str:
    """Build a concise per-step history of actions and their outcomes."""
    if not state.steps:
        return "  (no steps taken yet)"
    lines = []
    for step in state.steps:
        evidence_note = ""
        if step.result_count > 0:
            evidence_note = f", found {step.result_count} results"
        elif step.status == "error":
            evidence_note = ", FAILED"
        else:
            evidence_note = ", found nothing"
        lines.append(
            f"  Step {step.step}: {step.action}('{step.target[:60]}') -> {step.status}{evidence_note}"
        )
    return "\n".join(lines)


def _build_evidence_summary(state: AgentState) -> str:
    """Summarize collected evidence concisely."""
    if not state.candidate_evidence:
        return "None collected yet."
    snippets = []
    for ev in state.candidate_evidence[:4]:
        text_preview = ev["text"][:80].replace("\n", " ")
        snippets.append(f"  - [{ev['source']}] {text_preview}...")
    if len(state.candidate_evidence) > 4:
        snippets.append(f"  ... and {len(state.candidate_evidence) - 4} more items")
    return "\n".join(snippets)


class OrchestratorAgent:
    ALL_TOOLS = {"entity_linking", "vector_search", "graph_traversal", "document_retrieval", "stop"}

    def __init__(self, llm_client: LLMProvider):
        self.llm_client = llm_client

    def _get_blocked_tools(self, state: AgentState, max_tool_repeats: int = 2) -> set:
        """Determine which tools are exhausted/blocked."""
        blocked = set()
        for tool in self.ALL_TOOLS - {"stop"}:
            if state.tool_calls_count.get(tool, 0) >= max_tool_repeats:
                blocked.add(tool)
        return blocked

    async def decide_next_action(self, state: AgentState, eval_result: EvidenceEvalResult, timeout: float = None) -> OrchestratorAction:
        blocked = self._get_blocked_tools(state)
        available = sorted((self.ALL_TOOLS - blocked))

        prompt = ORCHESTRATOR_PROMPT.format(
            question=state.question,
            step_history=_build_step_history(state),
            evidence_count=len(state.candidate_evidence),
            eval_sufficient=eval_result.sufficient,
            eval_missing=", ".join(eval_result.missing_information) if eval_result.missing_information else "None",
            available_tools=", ".join(available)
        )

        try:
            resp = await self.llm_client.generate(
                prompt=prompt,
                format=OrchestratorAction.model_json_schema(),
                timeout=timeout,
                think=False
            )
            # Extract JSON block in case model includes filler text
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', resp, re.DOTALL)
            if match:
                clean_json = match.group(1)
            else:
                start_idx = resp.find('{')
                end_idx = resp.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    clean_json = resp[start_idx:end_idx+1]
                else:
                    clean_json = resp

            print(f"RAW JSON FROM MODEL (len={len(clean_json)}): {repr(clean_json)}", flush=True)
            logger.info("Raw JSON from model: %s", repr(clean_json))
            data = json.loads(clean_json)
            action = OrchestratorAction(**data)

            # Validate the chosen action is allowed
            if action.action not in self.ALL_TOOLS:
                action.action = "stop"

            # If the orchestrator chose a blocked tool, override
            if action.action in blocked:
                # Pick a different available tool
                non_stop = [t for t in available if t != "stop" and t != action.action]
                if non_stop:
                    logger.info("Orchestrator chose blocked tool '%s', redirecting to '%s'", action.action, non_stop[0])
                    action.action = non_stop[0]
                    action.reason = f"Redirected from blocked '{action.action}': {action.reason}"
                else:
                    action.action = "stop"
                    action.reason = "All tools exhausted, stopping."

            return action
        except Exception as e:
            logger.error(f"Orchestrator failed to decide: {e}")
            # Fallback: prefer vector_search to guarantee we get chunks, else pick any non-stop tool
            non_stop = [t for t in available if t != "stop"]
            fallback = "vector_search" if "vector_search" in available else (non_stop[0] if non_stop else "stop")

            return OrchestratorAction(
                action=fallback,
                target=state.question[:100],
                reason=f"Fallback due to orchestrator parse failure: {e}",
                fallback_used=True
            )
