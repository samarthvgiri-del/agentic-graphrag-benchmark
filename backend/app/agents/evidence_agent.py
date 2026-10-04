import json
from app.models.schemas import EvidenceEvalResult
from app.llm.base import LLMProvider
from app.agents.state import AgentState
from app.agents.context import ContextManager
import time

EVIDENCE_PROMPT = """You are an expert fact-checker evaluating evidence to answer a user's question.
Review the provided evidence and determine if it is sufficient to accurately and comprehensively answer the question.

Question: {question}

Evidence:
{context}

RULES:
1. If the evidence contains a direct answer (like a specific number, name, or event) to the question, set "sufficient": true.
2. Do not require exhaustive proof. If you have a solid, highly-likely answer based on the evidence, that is sufficient.
3. If no relevant evidence exists, set "sufficient": false.

Evaluate the evidence and output your response as valid JSON matching this schema:
{{
  "sufficient": boolean,
  "confidence": float (0.0 to 1.0),
  "missing_information": ["list", "of", "missing", "details"],
  "contradiction_detected": boolean,
  "recommended_next_action": "vector_search" | "entity_linking" | "graph_traversal" | "finish"
}}

Strictly adhere to the JSON schema.
"""

class EvidenceAgent:
    def __init__(self, llm_client: LLMProvider, context_manager: ContextManager):
        self.llm_client = llm_client
        self.context_manager = context_manager

    async def evaluate(self, state: AgentState) -> EvidenceEvalResult:
        context_str = self.context_manager.build_context_string(state)

        if not context_str.strip():
            return EvidenceEvalResult(
                sufficient=False,
                confidence=0.0,
                missing_information=["No evidence has been gathered yet."],
                contradiction_detected=False,
                recommended_next_action="vector_search"
            )

        # Lightweight deterministic checks first
        num_chunks = len(state.candidate_evidence)
        sources = set(ev.get("source", "unknown") for ev in state.candidate_evidence)
        source_diversity = len(sources)

        # Simple entity coverage check (deterministic)
        # Check if the words in the question appear in the evidence
        question_words = set(w.lower() for w in state.question.replace("?", "").replace(".", "").split() if len(w) > 4)
        evidence_text_lower = context_str.lower()

        matched_words = [w for w in question_words if w in evidence_text_lower]
        coverage = len(matched_words) / len(question_words) if question_words else 1.0

        if coverage < 0.3:
            return EvidenceEvalResult(
                sufficient=False,
                confidence=0.8,
                missing_information=["Question keywords are largely missing from evidence."],
                contradiction_detected=False,
                recommended_next_action="vector_search" if "vector_search" not in state.tools_called else "entity_linking"
            )

        # Fully deterministic checks without LLM call to save time and token costs
        # Check if relationships were found (if the question implies a relation)
        has_relationship = "Relationships:" in context_str

        is_sufficient = False
        next_action = "finish"

        if coverage >= 0.6 and num_chunks >= 2:
            is_sufficient = True
        elif coverage >= 0.4 and has_relationship:
            is_sufficient = True
        else:
            if "vector_search" not in state.tools_called:
                next_action = "vector_search"
            elif "entity_linking" not in state.tools_called:
                next_action = "entity_linking"
            elif "graph_traversal" not in state.tools_called and "entity_linking" in state.tools_called:
                next_action = "graph_traversal"
            else:
                next_action = "document_retrieval"

        # If we've done everything and still not sufficient, just finish
        if len(state.tools_called) >= 3 and not is_sufficient:
            next_action = "finish"
            # Just mark sufficient anyway to let it attempt generation
            is_sufficient = True

        return EvidenceEvalResult(
            sufficient=is_sufficient,
            confidence=coverage,
            missing_information=["Some entities might be missing" if coverage < 1.0 else "None"],
            contradiction_detected=False,
            recommended_next_action=next_action
        )
