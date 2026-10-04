from typing import List, Dict, Any
from app.agents.state import AgentState

class ContextManager:
    """
    Manages the context budget for the orchestrator to prevent LLM timeouts.
    """
    def __init__(self, max_items: int = 6, max_chars: int = 12000, max_tokens: int = 3000):
        self.max_items = max_items
        self.max_chars = max_chars
        self.max_tokens = max_tokens

    def build_context_string(self, state: AgentState) -> str:
        """
        Deduplicates, ranks, and truncates candidate evidence.
        Returns a single string ready for the LLM prompt.
        """
        # 1. Deduplicate by ID and exact text
        unique_texts = set()
        deduplicated = []
        for ev in state.candidate_evidence:
            text = ev["text"].strip()
            if text and text not in unique_texts:
                unique_texts.add(text)
                deduplicated.append(ev)

        # 2. Rank (Simplistic ranking: prioritize recent evidence or specific sources)
        # For this implementation, we just take the first N items,
        # or we could prioritize 'document_retrieval' over 'vector_search' if needed.
        # Let's just reverse to prioritize newest evidence found in the adaptive loop
        deduplicated.reverse()

        # 3. Select top K items
        selected = deduplicated[:self.max_items]

        # 4. Enforce character/token budget
        # We'll use a rough heuristic: 1 token ~ 4 chars
        char_budget = min(self.max_chars, self.max_tokens * 4)

        final_pieces = []
        current_chars = 0

        for ev in selected:
            text = f"[Source: {ev['id']}]\n{ev['text']}"
            if current_chars + len(text) > char_budget:
                remaining = char_budget - current_chars
                if remaining > 20:
                    final_pieces.append(text[:remaining] + "... [TRUNCATED]")
                break
            final_pieces.append(text)
            current_chars += len(text)

        return "\n\n".join(final_pieces)
