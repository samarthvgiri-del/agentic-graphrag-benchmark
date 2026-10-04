"""
Phase 4 — Graph ingestion script (batch-optimized).

Strategy:
  1. Parse the structured [Infobox Olympic event] deterministically (2934/2951 docs).
  2. Use LLM extraction with Ollama structured output ONLY for the ~17 docs
     that lack an infobox.
  3. All entities/relationships preserve full provenance:
     source document → source chunk → evidence text.
  4. Uses batch upserts to minimize network round-trips to TigerGraph Cloud.
"""

import sys
import os
import re
import json
import time
import logging
import asyncio
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict, Any, Tuple

from app.graph.tigergraph_client import TigerGraphClient
from app.llm.ollama_client import ollama_client
from app.rag.canonical_chunker import get_canonical_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Pydantic models for validation ──────────────────────────────────

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str

class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relationship_type: str
    evidence: str

class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]


# ── Ollama structured output JSON schema ────────────────────────────

EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {"type": "string"}
                },
                "required": ["name", "entity_type"]
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relationship_type": {"type": "string"},
                    "evidence": {"type": "string"}
                },
                "required": ["source", "target", "relationship_type", "evidence"]
            }
        }
    },
    "required": ["entities", "relationships"]
}

LLM_SYSTEM_PROMPT = """You are an information extraction system.
Extract entities and relationships from the provided text.

Allowed entity_type values: Athlete, Country, Event, Games, GenericEntity
Allowed relationship_type values: REPRESENTS, WON_MEDAL, COMPETED_IN, PART_OF, RELATES_TO

Rules:
- Only extract facts that are EXPLICITLY stated in the text.
- Do NOT invent or hallucinate any information.
- Each relationship MUST include the exact evidence text from the source.
- Return valid JSON matching the required schema."""


# ── Statistics tracker ──────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.docs_processed = 0
        self.docs_with_infobox = 0
        self.docs_without_infobox = 0
        self.chunks_attempted = 0
        self.chunks_successful = 0
        self.chunks_failed = 0
        self.entities_extracted = 0
        self.relationships_extracted = 0
        self.retries = 0
        self.rejected_relationships = 0

    def report(self):
        total = self.chunks_attempted or 1
        return {
            "documents_processed": self.docs_processed,
            "docs_with_infobox": self.docs_with_infobox,
            "docs_without_infobox": self.docs_without_infobox,
            "chunks_attempted": self.chunks_attempted,
            "chunks_successful": self.chunks_successful,
            "chunks_failed": self.chunks_failed,
            "extraction_success_rate": f"{self.chunks_successful / total * 100:.1f}%",
            "entities_extracted": self.entities_extracted,
            "relationships_extracted": self.relationships_extracted,
            "retries": self.retries,
            "rejected_relationships": self.rejected_relationships,
        }


# ── Batch accumulator ──────────────────────────────────────────────

class BatchAccumulator:
    """Accumulates vertices and edges, then flushes them in bulk."""

    FLUSH_SIZE = 200  # flush every N documents

    def __init__(self, client: TigerGraphClient):
        self.client = client
        # vertex_type -> list of (vertex_id, attributes_dict)
        self.vertices: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        # (source_type, edge_type, target_type) -> list of (source_id, target_id, attributes_dict)
        self.edges: Dict[Tuple[str, str, str], List[Tuple[str, str, Dict]]] = defaultdict(list)

    def add_vertex(self, vertex_type: str, vertex_id: str, attributes: Dict[str, Any]):
        self.vertices[vertex_type].append((vertex_id, attributes))

    def add_edge(self, source_type: str, edge_type: str, target_type: str,
                 source_id: str, target_id: str, attributes: Dict[str, Any] = None):
        self.edges[(source_type, edge_type, target_type)].append(
            (source_id, target_id, attributes or {})
        )

    def flush(self) -> Tuple[int, int]:
        """Flush all accumulated data to TigerGraph. Returns (vertices_upserted, edges_upserted)."""
        v_total = 0
        e_total = 0

        for vtype, vlist in self.vertices.items():
            if vlist:
                count = self.client.upsert_vertices_batch(vtype, vlist)
                v_total += count if isinstance(count, int) else 0

        for (stype, etype, ttype), elist in self.edges.items():
            if elist:
                count = self.client.upsert_edges_batch(stype, etype, ttype, elist)
                e_total += count if isinstance(count, int) else 0

        self.vertices.clear()
        self.edges.clear()
        return v_total, e_total


# ── Helpers ─────────────────────────────────────────────────────────

def normalize_entity_id(entity_type: str, name: str) -> str:
    norm = name.strip().lower().replace(" ", "_").replace("'", "").replace('"', '').replace(",", "")
    return f"{entity_type.lower()}:{norm}"


def parse_infobox(text: str) -> Optional[Dict[str, str]]:
    """Parse [Infobox ...] block from Wikipedia text into a dict of key:value pairs."""
    match = re.search(r'\[Infobox[^\]]*\](.*?)(?:\n\n|\Z)', text, re.DOTALL)
    if not match:
        return None
    block = match.group(1)
    fields = {}
    for line in block.strip().split('\n'):
        m = re.match(r'^\s+(\w+):\s*(.+)$', line)
        if m:
            fields[m.group(1).strip()] = m.group(2).strip()
    return fields if fields else None


VALID_ENTITY_TYPES = {"Athlete", "Country", "Event", "Games", "GenericEntity"}
VALID_REL_TYPES = {"REPRESENTS", "WON_MEDAL", "COMPETED_IN", "PART_OF", "RELATES_TO"}
EDGE_TYPE_CONSTRAINTS = {
    "REPRESENTS":  ("Athlete", "Country"),
    "WON_MEDAL":   ("Athlete", "Event"),
    "COMPETED_IN": ("Athlete", "Event"),
    "PART_OF":     ("Event", "Games"),
    "RELATES_TO":  ("GenericEntity", "GenericEntity"),
}


def extract_from_infobox(doc_id: str, title: str, fields: Dict[str, str], chunk_id: str):
    """Deterministically extract entities and relationships from structured infobox fields."""
    entities = []       # list of (type, name)
    relationships = []  # list of dicts

    event_name = fields.get("event", title)
    games_name = fields.get("games", "")

    if event_name:
        entities.append(("Event", event_name))
    if games_name:
        entities.append(("Games", games_name))
        if event_name:
            relationships.append({
                "source": event_name, "source_type": "Event",
                "target": games_name, "target_type": "Games",
                "rel_type": "PART_OF",
                "evidence": f"Infobox: event={event_name}, games={games_name}",
            })

    for medal_key, medal_type in [("gold", "Gold"), ("silver", "Silver"), ("bronze", "Bronze")]:
        for suffix in ["", "2", "3"]:
            key = f"{medal_key}{suffix}"
            noc_key = f"{medal_key}NOC{suffix}"
            athlete_raw = fields.get(key, "")
            country_code = fields.get(noc_key, "")

            if not athlete_raw:
                continue

            entities.append(("Athlete", athlete_raw))

            if country_code:
                entities.append(("Country", country_code))
                relationships.append({
                    "source": athlete_raw, "source_type": "Athlete",
                    "target": country_code, "target_type": "Country",
                    "rel_type": "REPRESENTS",
                    "evidence": f"Infobox: {key}={athlete_raw}, {noc_key}={country_code}",
                })

            if event_name:
                relationships.append({
                    "source": athlete_raw, "source_type": "Athlete",
                    "target": event_name, "target_type": "Event",
                    "rel_type": "WON_MEDAL",
                    "medal_type": medal_type,
                    "evidence": f"Infobox: {key}={athlete_raw}, event={event_name}",
                })

    return entities, relationships


# ── Defensive JSON parser ───────────────────────────────────────────

def defensive_parse_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip(), flags=re.MULTILINE)
    stripped = re.sub(r'\n?```\s*$', '', stripped.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── LLM extraction with retry ──────────────────────────────────────

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0

async def llm_extract_from_chunk(chunk_text: str, stats: Stats) -> ExtractionResult:
    prompt = f"Extract entities and relationships from this text:\n\n{chunk_text}"

    for attempt in range(MAX_RETRIES):
        try:
            response = await ollama_client.generate(
                prompt=prompt,
                system_prompt=LLM_SYSTEM_PROMPT,
                temperature=0.0,
                format=EXTRACTION_JSON_SCHEMA,
            )

            parsed = defensive_parse_json(response)
            if parsed is None:
                logger.warning(f"LLM returned unparseable response (attempt {attempt+1})")
                stats.retries += 1
                await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))
                continue

            result = ExtractionResult(**parsed)
            return result

        except ValidationError as e:
            logger.warning(f"Pydantic validation failed (attempt {attempt+1}): {e}")
            stats.retries += 1
            await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))
            continue

        except RuntimeError as e:
            error_msg = str(e)
            if "500" in error_msg or "Internal Server Error" in error_msg:
                logger.error(f"Ollama HTTP 500 on attempt {attempt+1}, retrying...")
            else:
                logger.error(f"Ollama error on attempt {attempt+1}: {error_msg}")
            stats.retries += 1
            await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))
            continue

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt+1}: {type(e).__name__}")
            stats.retries += 1
            await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))
            continue

    return ExtractionResult(entities=[], relationships=[])


# ── Entity / relationship insertion into batch ──────────────────────

def add_entities_and_relationships_to_batch(
    batch: BatchAccumulator,
    entities: list,
    relationships: list,
    chunk_id: str,
    doc_id: str,
    stats: Stats,
):
    entity_map = {}

    for ent_type, ent_name in entities:
        if ent_type not in VALID_ENTITY_TYPES:
            ent_type = "GenericEntity"

        ent_id = normalize_entity_id(ent_type, ent_name)
        entity_map[(ent_type, ent_name)] = ent_id

        attrs = {"name": ent_name}
        if ent_type == "GenericEntity":
            attrs["entity_type"] = "generic"

        batch.add_vertex(ent_type, ent_id, attrs)
        stats.entities_extracted += 1

        mention_edge = f"MENTIONS_{ent_type.upper()}" if ent_type != "GenericEntity" else "MENTIONS_GENERIC"
        batch.add_edge("Chunk", mention_edge, ent_type, chunk_id, ent_id)

    for rel in relationships:
        s_name = rel.get("source", "")
        s_type = rel.get("source_type", "")
        t_name = rel.get("target", "")
        t_type = rel.get("target_type", "")
        rel_type = rel.get("rel_type") or rel.get("relationship_type", "")

        if not all([s_name, s_type, t_name, t_type, rel_type]):
            stats.rejected_relationships += 1
            continue

        if rel_type not in VALID_REL_TYPES:
            stats.rejected_relationships += 1
            continue

        expected = EDGE_TYPE_CONSTRAINTS.get(rel_type)
        if expected and (s_type, t_type) != expected:
            stats.rejected_relationships += 1
            continue

        s_id = entity_map.get((s_type, s_name)) or normalize_entity_id(s_type, s_name)
        t_id = entity_map.get((t_type, t_name)) or normalize_entity_id(t_type, t_name)

        edge_attrs = {"source_chunk_id": chunk_id}
        if rel_type == "WON_MEDAL":
            edge_attrs["medal_type"] = rel.get("medal_type", "")
        elif rel_type == "RELATES_TO":
            edge_attrs["description"] = rel.get("evidence", rel.get("description", ""))

        batch.add_edge(s_type, rel_type, t_type, s_id, t_id, edge_attrs)
        stats.relationships_extracted += 1


# ── Main build function ────────────────────────────────────────────

async def build_graph(corpus_path: str):
    stats = Stats()

    client = TigerGraphClient()
    if not client.configured:
        logger.error("TigerGraph is not configured. Set TG env vars in .env")
        return

    status = client.get_status()
    if status["status"] != "connected":
        logger.error(f"Cannot connect to TigerGraph: {status}")
        return

    logger.info("Connected to TigerGraph. Initializing schema...")
    client.init_schema()
    await asyncio.sleep(3)

    batch = BatchAccumulator(client)

    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_docs = len(lines)
    logger.info(f"Loaded {total_docs} documents from corpus.")

    logger.info("Precomputing canonical chunks...")
    all_chunks = get_canonical_chunks(corpus_path)
    precomputed_chunks = defaultdict(list)
    for c in all_chunks:
        precomputed_chunks[c.document_id].append(c)
    logger.info(f"Precomputed {len(all_chunks)} canonical chunks across {len(precomputed_chunks)} documents.")

    start_time = time.time()

    for doc_idx, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON at line {doc_idx}")
            continue

        doc_id = str(doc["doc_id"])
        title = doc.get("title", "")
        url = doc.get("url", "")
        text = doc.get("text", "")

        # Document vertex
        batch.add_vertex("Document", doc_id, {"title": title, "url": url})

        # Get canonical chunks for this document
        doc_chunks = precomputed_chunks.get(doc_id, [])

        # Add chunks
        for chunk in doc_chunks:
            chunk_id = chunk.chunk_id
            batch.add_vertex("Chunk", chunk_id, {"text": chunk.text})
            batch.add_edge("Document", "HAS_CHUNK", "Chunk", doc_id, chunk_id)
            stats.chunks_attempted += 1
            stats.chunks_successful += 1

        first_chunk_id = doc_chunks[0].chunk_id if doc_chunks else doc_id

        # Parse infobox (deterministic extraction)
        infobox = parse_infobox(text)

        if infobox:
            stats.docs_with_infobox += 1
            infobox_entities, infobox_relationships = extract_from_infobox(
                doc_id, title, infobox, first_chunk_id
            )
            add_entities_and_relationships_to_batch(
                batch, infobox_entities, infobox_relationships,
                first_chunk_id, doc_id, stats
            )
        else:
            stats.docs_without_infobox += 1
            # LLM extraction for non-infobox docs
            for chunk in doc_chunks:
                chunk_id = chunk.chunk_id
                chunk_text = chunk.text

                logger.info(f"LLM extracting from chunk {chunk_id} (doc {doc_id})...")

                try:
                    result = await llm_extract_from_chunk(chunk_text, stats)

                    if not result.entities and not result.relationships:
                        stats.chunks_failed += 1
                        continue

                    ents = [(e.entity_type if e.entity_type in VALID_ENTITY_TYPES else "GenericEntity", e.name)
                            for e in result.entities]
                    rels = []
                    for r in result.relationships:
                        rel_dict = {
                            "source": r.source,
                            "source_type": "GenericEntity",
                            "target": r.target,
                            "target_type": "GenericEntity",
                            "rel_type": r.relationship_type if r.relationship_type in VALID_REL_TYPES else "RELATES_TO",
                            "evidence": r.evidence,
                            "description": r.evidence,
                        }
                        rels.append(rel_dict)

                    add_entities_and_relationships_to_batch(
                        batch, ents, rels, chunk_id, doc_id, stats
                    )

                except Exception as e:
                    logger.error(f"Chunk {chunk_id} failed: {type(e).__name__}: {e}")
                    stats.chunks_failed += 1
                    continue

        stats.docs_processed += 1

        # Flush periodically
        if (doc_idx + 1) % BatchAccumulator.FLUSH_SIZE == 0:
            logger.info(f"Flushing batch at doc {doc_idx + 1}/{total_docs}...")
            v, e = batch.flush()
            logger.info(f"  Flushed {v} vertices, {e} edges")
            elapsed = time.time() - start_time
            rate = (doc_idx + 1) / elapsed
            logger.info(f"  Progress: {doc_idx + 1}/{total_docs} docs ({rate:.1f} docs/sec)")

    # Final flush
    logger.info("Final flush...")
    v, e = batch.flush()
    logger.info(f"  Flushed {v} vertices, {e} edges")

    elapsed = time.time() - start_time

    # Final report
    logger.info("=" * 60)
    logger.info("GRAPH BUILD COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 60)

    report = stats.report()
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    # Fetch TigerGraph counts
    try:
        v_counts = client.client.getVertexCount('*')
        e_counts = client.client.getEdgeCount('*')
        logger.info("  --- TigerGraph Vertex Counts ---")
        for vt, count in v_counts.items():
            logger.info(f"    {vt}: {count}")
        logger.info("  --- TigerGraph Edge Counts ---")
        for et, count in e_counts.items():
            logger.info(f"    {et}: {count}")
    except Exception as e:
        logger.error(f"  Could not fetch TigerGraph counts: {e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus", "corpus.jsonl")
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found: {corpus_path}")
        sys.exit(1)
    asyncio.run(build_graph(corpus_path))
