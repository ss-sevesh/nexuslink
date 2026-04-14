# NexusLink Architecture

## Overview

NexusLink is a three-layer pipeline that transforms raw scientific literature into ranked, novel research hypotheses.

```
PDFs / ArXiv / DOI
        │
        ▼
┌─────────────────────────────────────┐
│  Layer 1: RAW                       │
│  ingestion → extraction → store     │
│  Output: structured JSON + vectors  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Layer 2: WIKI                      │
│  graph ← linker → taxonomy          │
│              ↓                      │
│           citations                 │
│  Output: queryable knowledge graph  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Layer 3: LLM                       │
│  hypothesis → scoring → reports     │
│  Output: ranked hypothesis report   │
└─────────────────────────────────────┘
```

## Data Flow

### RAW Layer

1. **ingestion/loader.py** accepts a PDF path, ArXiv ID, or DOI.
2. PDFs are converted to Markdown via `pymupdf4llm`. ArXiv papers fetched via the `arxiv` client.
3. **extraction/ner.py** runs spaCy + sciSpaCy over the Markdown to extract entities.
4. **extraction/claims.py** identifies claim sentences using dependency parsing.
5. **store/vector.py** embeds text chunks with `sentence-transformers` and upserts into ChromaDB.
6. **store/document.py** serialises the full `RawDocument` to JSON on disk.

### WIKI Layer

1. **graph/builder.py** loads all `RawDocument` JSON files and constructs a NetworkX DiGraph.
   - Nodes: `(entity_id, {type, label, domain, embedding})`
   - Edges: `(src, dst, {relation, weight, source_doc})`
2. **linker/bridge.py** queries ChromaDB for top-K nearest neighbours across domain boundaries.
   - A bridge is created when two entities from different taxonomy domains exceed a similarity threshold and share no direct citation path.
3. **taxonomy/mapper.py** assigns each entity to a domain tree using ArXiv category codes and MESH terms.
4. **citations/manager.py** resolves DOIs via `habanero`, builds BibTeX records, and tracks claim-citation mappings.

### LLM Layer

1. **hypothesis/generate.py** iterates over bridges from the WIKI layer.
   - For each bridge, renders a Jinja2 prompt from `prompts/hypothesis.j2`.
   - Calls Claude via the `anthropic` SDK; parses structured output into `Hypothesis` models.
2. **scoring/ranker.py** scores each hypothesis on four axes (novelty, feasibility, impact, cross-domain span) and produces a ranked list.
3. **validation/checker.py** embeds each hypothesis claim and searches ChromaDB for contradicting evidence.
4. **reports/generator.py** renders the final ranked list into a Markdown or LaTeX report with a full bibliography.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ChromaDB over Pinecone | Local-first, no API key required for development |
| NetworkX over Neo4j | Pure Python, easier to ship; Neo4j adapter is a future option |
| Pydantic v2 throughout | Fast validation, JSON schema generation for API layer |
| Jinja2 prompts | Keeps prompt logic version-controlled and testable independently of Python code |
| litellm wrapper | Swap model providers without changing hypothesis generation code |

## Extension Points

- **New ingestion source**: implement `BaseLoader` in `raw/ingestion/base.py`, register in `loader.py`.
- **New relation type**: add to `RelationType` enum in `wiki/graph/schema.py`; bridge detector picks it up automatically.
- **New scoring axis**: subclass `BaseScoringAxis` in `llm/scoring/base.py` and register in `ranker.py`.
