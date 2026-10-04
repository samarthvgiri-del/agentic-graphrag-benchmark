# Graph Schema

This document defines the TigerGraph schema used for Phase 4 GraphRAG integration. The schema is based on the actual `data/corpus/corpus.jsonl` which contains Wikipedia articles about Olympic events, athletes, and medals.

## Design Philosophy
We combine strict semantic types for common entities with a clear provenance trail. Generic entities are used as a fallback to avoid data loss. Every fact (edge) traces back to a specific text chunk.

## Vertices (Nodes)

### Source Provenance
- `Document`
  - `id` (PRIMARY KEY, STRING): e.g., "Q303623"
  - `title` (STRING)
  - `url` (STRING)
- `Chunk`
  - `id` (PRIMARY KEY, STRING): e.g., "Q303623_chunk_0"
  - `text` (STRING): The actual text content

### Semantic Entities
- `Athlete`
  - `id` (PRIMARY KEY, STRING): e.g., "athlete:rudolf_dombi"
  - `name` (STRING)
- `Country`
  - `id` (PRIMARY KEY, STRING): e.g., "country:hun"
  - `name` (STRING)
- `Event`
  - `id` (PRIMARY KEY, STRING): e.g., "event:mens_canoe_sprint_k-2_1000_metres"
  - `name` (STRING)
- `Games`
  - `id` (PRIMARY KEY, STRING): e.g., "games:2012_summer"
  - `name` (STRING)
- `GenericEntity` *(Fallback)*
  - `id` (PRIMARY KEY, STRING)
  - `name` (STRING)
  - `entity_type` (STRING)

## Edges (Relationships)

### Provenance Edges
- `HAS_CHUNK` (From `Document` to `Chunk`)
- `MENTIONS_ATHLETE` (From `Chunk` to `Athlete`)
- `MENTIONS_COUNTRY` (From `Chunk` to `Country`)
- `MENTIONS_EVENT` (From `Chunk` to `Event`)
- `MENTIONS_GAMES` (From `Chunk` to `Games`)
- `MENTIONS_GENERIC` (From `Chunk` to `GenericEntity`)

### Semantic Relationship Edges
*All relationships include a `source_chunk_id` attribute for strict provenance.*

- `REPRESENTS` (From `Athlete` to `Country`)
  - Attributes: `source_chunk_id` (STRING)
- `WON_MEDAL` (From `Athlete` to `Event`)
  - Attributes: `medal_type` (STRING) e.g., "Gold", "Silver", "Bronze", `source_chunk_id` (STRING)
- `COMPETED_IN` (From `Athlete` to `Event`)
  - Attributes: `source_chunk_id` (STRING)
- `PART_OF` (From `Event` to `Games`)
  - Attributes: `source_chunk_id` (STRING)

### Fallback Relationship Edge
- `RELATES_TO` (From `GenericEntity` to `GenericEntity`)
  - Attributes: `description` (STRING), `source_chunk_id` (STRING)
