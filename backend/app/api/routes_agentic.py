import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)
from app.models.schemas import RAGQueryRequest, AgenticQueryResponse, AgenticTrace
from app.agents.state import AgentState
from app.agents.context import ContextManager
from app.agents.evidence_agent import EvidenceAgent
from app.agents.stopping import StoppingPolicy
from app.agents.orchestrator import OrchestratorAgent
from app.agents.tools import AgenticTools
from app.api._shared_store import get_vector_store
from app.llm.ollama_client import OllamaClient
from app.graph.tigergraph_client import TigerGraphClient
from app.embeddings.embedder import embedder
from app.graph.graphrag_pipeline import ANSWER_PROMPT
import time

router = APIRouter(prefix="/api/query", tags=["agentic"])

@router.post("/agentic", response_model=AgenticQueryResponse)
async def agentic_query(request: RAGQueryRequest):
    """Run an Adaptive Agentic GraphRAG query."""
    start_time = time.time()

    # Initialize components
    from app.llm.factory import get_llm_provider, get_orchestrator_provider, get_evaluator_provider
    llm_client = get_llm_provider()
    orchestrator_client = get_orchestrator_provider()
    evaluator_client = get_evaluator_provider()
    tg_client = TigerGraphClient()
    qdrant_client = get_vector_store()

    tools = AgenticTools(tg_client, qdrant_client, embedder)
    context_manager = ContextManager(max_items=6, max_chars=12000, max_tokens=3000)
    evidence_agent = EvidenceAgent(evaluator_client, context_manager)
    orchestrator = OrchestratorAgent(orchestrator_client)
    from app.config import settings
    stopping_policy = StoppingPolicy(
        max_steps=settings.AGENT_MAX_STEPS,
        max_tool_repeats=settings.AGENT_MAX_TOOL_REPEATS,
        max_runtime_ms=settings.AGENT_RUNTIME_BUDGET_MS
    )

    state = AgentState(request.question)

    while not state.is_finished:
        # 1. Evaluate Evidence
        eval_result = await evidence_agent.evaluate(state)

        # 2. Check Stopping Policy
        if stopping_policy.should_stop(state, eval_result):
            break

        # Check overall runtime budget
        elapsed_ms = (time.time() - start_time) * 1000
        remaining_ms = settings.AGENT_RUNTIME_BUDGET_MS - elapsed_ms
        if remaining_ms <= 0:
            state.set_finished(f"Maximum runtime budget reached ({elapsed_ms:.0f}ms).")
            break

        request_timeout = min(settings.OLLAMA_REQUEST_TIMEOUT_MS / 1000.0, remaining_ms / 1000.0)

        # 3. Decide Next Action (with retry if blocked)
        orch_start = time.time()
        action = await orchestrator.decide_next_action(state, eval_result, timeout=request_timeout)
        logger.info(f"Orchestrator decided '{action.action}' in {(time.time() - orch_start)*1000:.0f}ms")

        if action.action == "stop":
            state.set_finished("Orchestrator explicitly chose to finish.")
            break

        # 3b. Check if this action is blocked; if so, ask orchestrator again
        block_reason = stopping_policy.should_block_action(state, action)
        if block_reason:
            logger.info("Action blocked: %s. Marking as failed.", block_reason)
            state.failed_actions.add(f"{action.action}:{action.target}")

            # Recalculate remaining timeout
            elapsed_ms = (time.time() - start_time) * 1000
            remaining_ms = settings.AGENT_RUNTIME_BUDGET_MS - elapsed_ms
            if remaining_ms <= 0:
                state.set_finished(f"Maximum runtime budget reached ({elapsed_ms:.0f}ms).")
                break
            request_timeout = min(settings.OLLAMA_REQUEST_TIMEOUT_MS / 1000.0, remaining_ms / 1000.0)

            # Re-ask orchestrator with updated state
            action = await orchestrator.decide_next_action(state, eval_result, timeout=request_timeout)
            if action.action == "stop":
                state.set_finished("Orchestrator chose to finish after blocked action.")
                break
            # Check again
            block_reason2 = stopping_policy.should_block_action(state, action)
            if block_reason2:
                state.set_finished(f"All viable tools exhausted: {block_reason2}")
                break

        # 4. Execute Action
        result_summary = ""
        tool_latency = 0
        result = None

        if action.action == "vector_search":
            state.record_strategy_change("Orchestrator selected vector_search", "vector_search")
            result = tools.vector_search(action.target)
            state.add_tool_result(action, result)
            result_summary = f"Found {len(result.results)} vectors."
            tool_latency = result.latency_ms

        elif action.action == "entity_linking":
            state.record_strategy_change("Orchestrator selected entity_linking", "entity_linking")
            result = tools.entity_linking(action.target)
            state.add_tool_result(action, result)
            result_summary = f"Linked {len(result.results)} entities."
            tool_latency = result.latency_ms

        elif action.action == "graph_traversal":
            state.record_strategy_change("Orchestrator selected graph_traversal", "graph_traversal")
            result = tools.graph_traversal(action.target)
            state.add_tool_result(action, result)
            result_summary = f"Traversed and found {len(result.results)} chunks/relationships."
            tool_latency = result.latency_ms

        elif action.action == "document_retrieval":
            state.record_strategy_change("Orchestrator selected document_retrieval", "document_retrieval")
            result = tools.document_retrieval(action.target)
            state.add_tool_result(action, result)
            result_summary = f"Retrieved {len(result.results)} document chunks."
            tool_latency = result.latency_ms

        else:
            result_summary = f"Unknown action: {action.action}"

        # 5. Record Step
        status_str = "success" if (result and result.success) else "error"
        input_size = len(action.target) if action.target else 0
        output_size = sum(len(ev) for ev in result.evidence) if result and result.evidence else 0
        result_count = len(result.results) if result and result.results else 0
        candidate_count = result_count  # simplified approximation
        think_enabled = False
        from app.config import settings
        timeout_val = settings.OLLAMA_REQUEST_TIMEOUT_MS / 1000.0

        state.record_step(
            action=action,
            latency_ms=tool_latency,
            status=status_str,
            input_size=input_size,
            output_size=output_size,
            candidate_count=candidate_count,
            result_count=result_count,
            think_enabled=think_enabled,
            timeout_val=timeout_val
        )

    # 6. Generate Final Answer
    context_str = context_manager.build_context_string(state)
    final_prompt = ANSWER_PROMPT.format(context=context_str, question=state.question)

    elapsed_ms = (time.time() - start_time) * 1000
    remaining_ms = settings.AGENT_RUNTIME_BUDGET_MS - elapsed_ms

    answer_start = time.time()
    if remaining_ms <= 0:
        answer = f"Failed to generate answer: Maximum runtime budget reached before generation ({elapsed_ms:.0f}ms)."
        state.set_finished("Budget exhausted before answer generation.")
    else:
        answer_timeout = min(settings.OLLAMA_REQUEST_TIMEOUT_MS / 1000.0, remaining_ms / 1000.0)
        try:
            answer = await llm_client.generate(
                prompt=final_prompt,
                temperature=0.2,
                think=False,
                num_predict=800,
                timeout=answer_timeout
            )
        except Exception as e:
            answer = f"Failed to generate answer: {str(e)}"
    answer_latency = int((time.time() - answer_start) * 1000)

    total_latency_ms = int((time.time() - start_time) * 1000)

    trace = AgenticTrace(
        question=state.question,
        steps=state.steps,
        tools_called=state.tools_called,
        strategy_changes=state.strategy_changes,
        evidence_count=len(state.candidate_evidence),
        citations=list(state.seen_evidence_ids),
        stop_reason=state.stop_reason or "Unknown",
        total_steps=len(state.steps),
        total_latency_ms=state.total_latency_ms
    )

    return AgenticQueryResponse(
        answer=answer,
        citations=list(state.seen_evidence_ids),
        trace=trace,
        entities=state.entities,
        relationships=state.relationships,
        retrieved_chunks=state.candidate_evidence,
        total_steps=len(state.steps),
        latency_ms=total_latency_ms,
        stop_reason=state.stop_reason or "Unknown"
    )
