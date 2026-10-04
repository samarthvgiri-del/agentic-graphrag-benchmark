import time
import json
import logging
import re
from typing import Dict, Any

from app.llm.ollama_client import ollama_client
from app.graph.tigergraph_client import TigerGraphClient
from app.graph.entity_linker import EntityLinker
from app.graph.graph_traversal import GraphTraversal
from app.models.schemas import EntityExtraction

logger = logging.getLogger(__name__)

# Prompt to extract entities from the question
EXTRACT_PROMPT = """Extract the key entities from this question to query a knowledge graph.
Allowed types: Athlete, Country, Event, Games, GenericEntity
Return ONLY valid JSON: {{"entities": [{{"name": "...", "type": "..."}}]}}

IMPORTANT: OUTPUT ONLY THE JSON OBJECT AND NOTHING ELSE. DO NOT OUTPUT ANY PREAMBLE OR THINKING TEXT. YOU MUST START YOUR RESPONSE WITH '{{' AND END WITH '}}'.

Example Question: Who won the men's 100m sprint in the 2012 London Olympics?
Example Output: {{"entities": [{{"name": "men's 100m sprint", "type": "Event"}}, {{"name": "2012 London Olympics", "type": "Games"}}]}}

Question: {question}
Output:
"""

# Prompt to answer using context
ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
Context includes both structured relationships from a knowledge graph and the original text chunks.
Cite your sources by including the [chunk_id] when you use information from a chunk.
If the context does not contain enough information, say "Insufficient evidence to answer the question." Do not invent facts.

Context:
{context}

Question:
{question}
"""

async def run_graphrag(question: str, client: TigerGraphClient, vector_store=None) -> Dict[str, Any]:
    start_time = time.time()

    linker = EntityLinker(client, vector_store=vector_store)
    traverser = GraphTraversal(client)

def extract_entities_deterministic(question: str) -> List[Dict[str, Any]]:
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
    seasons = [s for s in ["Summer", "Winter"] if s in question]

    phrases = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', question)

    months = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
    stop_words = {
        "Who", "What", "When", "Where", "Why", "How", "Which", "Whom", "Whose",
        "According", "In", "On", "At", "By", "To", "From", "With", "About",
        "The", "A", "An", "Is", "Are", "Was", "Were", "Has", "Have", "Had",
        "Do", "Does", "Did", "Can", "Could", "Should", "Would", "May", "Might", "Must",
        "held", "immediately", "before", "after", "during", "event", "events", "number", "medal", "gold", "silver", "bronze"
    } | months

    query_entities = []

    if years:
        for y in years:
            for s in (seasons if seasons else ["Summer", "Winter"]):
                query_entities.append({"name": f"{y} {s}", "type": "Games"})

    for p in phrases:
        p_clean = p.strip()
        if p_clean in ["Summer Olympics", "Winter Olympics"]:
            continue

        first_word = p_clean.split()[0]
        if first_word in stop_words or p_clean in stop_words:
            continue

        if len(p_clean) > 2 and not p_clean.isdigit():
            e_type = "GenericEntity"
            lower_q = question.lower()
            if "athlete" in lower_q or "won" in lower_q:
                pass
            if "event" in lower_q or "competitors" in lower_q or "sprint" in lower_q or "relay" in lower_q:
                pass
            query_entities.append({"name": p_clean, "type": "GenericEntity"})

    seen = set()
    unique_entities = []
    for e in query_entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            unique_entities.append(e)

    return unique_entities

async def run_graphrag(question: str, client: TigerGraphClient, vector_store=None) -> Dict[str, Any]:
    start_time = time.time()

    linker = EntityLinker(client, vector_store=vector_store)
    traverser = GraphTraversal(client)

    # 1. Deterministic entity extraction from question
    query_entities = extract_entities_deterministic(question)

    logger.info(f"[GraphRAG] Q: {question[:50]}... | Extracted {len(query_entities)} entities: {[e.get('name') for e in query_entities]}")

    # 2. Link entities
    linked_entities = linker.link_entities(query_entities)

    logger.debug(f"[GraphRAG] Linked entities detail: {json.dumps(linked_entities)}")
    if linked_entities:
        logger.info(f"[GraphRAG] Linked {len(linked_entities)} candidates. Selected IDs: {[e['linked_id'] for e in linked_entities]}")
    else:
        logger.info(f"[GraphRAG] Failed to link any candidates.")

    # 3. Traverse Graph
    traversal_results = traverser.traverse(linked_entities)
    relationships = traversal_results.get("relationships", [])
    chunks = traversal_results.get("retrieved_chunks", [])

    logger.info(f"[GraphRAG] Traversal complete. Relationships: {len(relationships)}, Chunks: {len(chunks)}")
    logger.debug(f"[GraphRAG] Traversal supporting chunks: {[c.get('chunk_id') for c in chunks]}")

    # 4. Format context
    context_str = "Relationships:\n"
    for r in relationships:
        attr_str = str(r.get("attributes", {}))
        context_str += f"- {r['source']} -> {r['rel_type']} -> {r['target']} {attr_str}\n"

    context_str += "\nSource Chunks:\n"
    for c in chunks:
        context_str += f"[{c['chunk_id']}] {c['text']}\n"

    from app.llm.factory import get_llm_provider
    llm_client = get_llm_provider()

    # 5. Generate Answer
    answer = await llm_client.generate(
        prompt=ANSWER_PROMPT.format(context=context_str, question=question),
        temperature=0.2,
        think=False,
        num_predict=800
    )

    latency = int((time.time() - start_time) * 1000)
    logger.info(f"[GraphRAG] Pipeline complete in {latency}ms. Answer length: {len(answer)}")

    return {
        "answer": answer,
        "citations": [c["chunk_id"] for c in chunks],
        "entities": linked_entities,
        "relationships": relationships,
        "retrieved_chunks": chunks,
        "latency_ms": latency,
        "extracted_phrases": list(set(phrases)),
        "linked_entities": [e["linked_id"] for e in linked_entities],
        "graph_nodes_returned": len(relationships) * 2,
        "retrieved_chunk_count": len(chunks),
        "unique_source_count": len(set(c.get("source", "unknown") for c in chunks))
    }
