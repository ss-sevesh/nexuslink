# NexusLink — Self-Refining Cross-Domain Hypothesis Engine with Evidence Integrity

## What This Project Does

NexusLink ingests research papers from multiple scientific domains, builds a cross-domain knowledge graph inside an Obsidian vault, generates novel hypotheses by identifying non-obvious conceptual bridges between fields, verifies the integrity of all evidence chains (retraction checking, citation validation), and improves with every cycle through human feedback and autonomous paper discovery.

## Key Differentiator

**Unlike one-shot systems (SciAgents, ResearchLink, KG-CoI), NexusLink is cyclical and self-refining.**

| Feature | SciAgents | ResearchLink | KG-CoI | NexusLink |
|---------|-----------|--------------|--------|-----------|
| KG → Hypothesis | ✓ | ✓ | ✓ | ✓ |
| Cyclical Refinement | ✗ | ✗ | ✗ | ✓ |
| Human-in-the-Loop | ✗ | ✗ | ✗ | ✓ |
| Self-Healing Vault | ✗ | ✗ | ✗ | ✓ |
| Autonomous Expansion | ✗ | ✗ | ✗ | ✓ |
| Evidence Integrity | ✗ | ✗ | ✗ | ✓ |
| Obsidian Interface | ✗ | ✗ | ✗ | ✓ |

## Architecture: Cyclical Pipeline

```
Ingest → Vault → Heal → Feedback → Link → Hypothesize → Integrity Check → Expand → Vault (repeat)
```

Each cycle:

- **READ** — VaultReader scans all notes, detects human edits
- **HEAL** — Merge duplicate concepts, fix broken links, prune junk, propagate human corrections
- **FEEDBACK** — Collect reviewed/rejected hypotheses as few-shot examples, calibrate scoring
- **LINK** — Embed concepts, find cross-domain bridges (domain distance ≥ 2), rerank with cross-encoder
- **HYPOTHESIZE** — Generate with mechanistic reasoning + falsifiable predictions, inject feedback
- **INTEGRITY** — Check retraction status via CrossRef, score evidence reliability via Semantic Scholar
- **EXPAND** — Auto-search for supporting/refuting papers, ingest into vault
- **RECORD** — Log cycle stats, write cycle note

## Layer Details

### RAW (raw/)
- PDF ingestion: pymupdf4llm
- ArXiv: arxiv API client
- DOI: habanero (CrossRef)
- NER: spaCy/sciSpaCy + 200 phrase patterns
- Output: `wiki/01-papers/*.md` (Obsidian notes with YAML + `[[entities]]`)

### WIKI (wiki/)
- The `wiki/` folder IS an Obsidian vault — open it in Obsidian
- Embeddings: all-mpnet-base-v2 (768-dim) with context sentences
- Bridge finding: cross-domain only (domain distance ≥ 2), co-occurrence filter, near-duplicate filter
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2
- Knowledge graph: NetworkX DiGraph
- Output: `wiki/02-concepts/*.md` with `[[wikilinks]]`

### LLM (llm/)
- Claude Sonnet (default) or Ollama (fallback)
- Enforces: Mechanism A + Mechanism B + Connection Rationale + Falsifiable Prediction
- Scoring: novelty, feasibility, impact, mechanistic_depth, falsifiability
- Calibrated by human feedback over cycles
- Output: `wiki/03-hypotheses/*.md` with full structure

### Vault System (wiki/vault/)
- `reader.py` — Parses all vault notes, detects human edits via content hashing
- `healer.py` — Merges duplicates, fixes broken links, prunes low-quality, propagates human corrections
- `feedback.py` — Builds few-shot examples from reviewed/rejected hypotheses, calibrates scoring
- `expander.py` — Searches Semantic Scholar for supporting/refuting papers, auto-ingests
- `integrity.py` — Checks retractions via CrossRef, scores evidence reliability, flags compromised hypotheses

## Tech Stack
- **Runtime**: Python 3.12, uv
- **Core**: pydantic v2, networkx, sentence-transformers, chromadb
- **PDF**: pymupdf4llm, arxiv, habanero
- **NLP**: spaCy, sciSpaCy
- **LLM**: anthropic SDK (Claude), ollama (fallback)
- **APIs**: Semantic Scholar, CrossRef (retraction data), Unpaywall
- **API**: FastAPI + uvicorn
- **Testing**: pytest
- **Linting**: ruff

## CLI Commands
```
nexuslink ingest <source>          # Ingest PDF/ArXiv/DOI
nexuslink ingest-batch <folder>    # Ingest all PDFs in folder
nexuslink link                     # Run cross-domain linking
nexuslink hypothesize              # Generate hypotheses
nexuslink cycle                    # Run one full cycle (the main command)
nexuslink cycle --continuous       # Run cycles until vault stabilizes
nexuslink heal                     # Run vault healer (--apply to execute)
nexuslink integrity                # Check evidence integrity of all hypotheses
nexuslink expand                   # Auto-expand vault with new papers
nexuslink feedback                 # Show feedback/calibration state
nexuslink benchmark --cycles 3     # Run one-shot vs cyclic comparison
nexuslink benchmark --export-latex # Export LaTeX table for paper
nexuslink status                   # Show vault stats
```

## How Scientists Use This

1. `nexuslink ingest` — feed it papers from 3+ different domains
2. `nexuslink cycle` — run first cycle
3. Open `wiki/` in Obsidian — browse graph view, read hypotheses
4. Edit hypothesis YAML: change `status` to `"reviewed"` or `"rejected"`, adjust scores, add Human Notes
5. `nexuslink cycle` — second cycle learns from your edits
6. Repeat — each cycle: better hypotheses, verified evidence, growing vault

## Resume Context
- Check PROGRESS.md for current phase
- Each cycle is independently runnable
- The vault is the source of truth — always start by reading it

## Prior Work
- SciAgents (Ghafarollahi & Buehler, 2024) — multi-agent KG hypothesis generation (one-shot)
- ResearchLink (2025) — link prediction over KGs for hypothesis generation
- KG-CoI (2024) — KG-augmented chain-of-idea with hallucination detection
- CrossTrace (2025) — dataset of cross-domain reasoning traces
- Zhao et al. (2009) — repulsive Casimir force in chiral metamaterials (foundational reference for bridge concept)

## Funding Targets
- NSF Convergence Accelerator (AI4Science track)
- DARPA AIE
- Wellcome Trust Discovery Award
