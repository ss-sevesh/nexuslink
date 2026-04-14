# NexusLink Progress Tracker

## Phase 1: RAW Layer (Current)
- [x] PDF ingestion pipeline — `raw/ingestion/pdf_loader.py`
- [x] ArXiv API integration — `raw/ingestion/arxiv_loader.py`
- [x] Entity extraction (spaCy + sciSpaCy) — `raw/extraction/entity_extractor.py`
- [x] Pydantic schemas finalized — `raw/schemas/models.py`
- [x] Pipeline orchestrator with wiki note writer — `raw/ingestion/pipeline.py`
- [x] Tests — `tests/test_raw/test_pipeline.py`
- [ ] ChromaDB vector store setup
- [ ] Citation parser (BibTeX/DOI)

## Phase 2: WIKI Layer (Current)
- [x] Knowledge graph schema (NetworkX) — `wiki/graph/builder.py`
- [x] Cross-domain linker v1 — `wiki/linker/bridge_finder.py`
- [x] Concept embedder — `wiki/linker/embedder.py`
- [x] Linking pipeline (vault reader + bridge exporter) — `wiki/linker/pipeline.py`
- [x] Taxonomy mapping — `wiki/taxonomy/classifier.py`
- [x] Citation manager (BibTeX + CrossRef) — `wiki/citations/manager.py`
- [x] Tests — `tests/test_wiki/test_linker.py`
- [ ] WikiLink backlink graph (obsidiantools integration deeper)

## Phase 3: LLM Layer (Built)
- [x] Hypothesis generation prompts — `llm/prompts/templates.py`
- [x] Hypothesis generator (Claude API, batched) — `llm/hypothesis/generator.py`
- [x] Novelty scoring algorithm (composite 0.4N+0.3I+0.3F) — `llm/scoring/ranker.py`
- [x] Claim validation + contradiction detection — `llm/validation/checker.py`
- [x] Report generator (LaTeX + MD, wiki/04-reports/) — `llm/reports/writer.py`
- [x] End-to-end pipeline — `llm/hypothesis/pipeline.py`
- [x] Tests — `tests/test_llm/test_hypothesis.py`

## Phase 4: Integration & API
- [ ] FastAPI endpoints
- [ ] End-to-end pipeline
- [ ] Demo with 3 domains

## Phase 5: Publication & Funding
- [ ] Paper draft (target: AAAI/NeurIPS workshop)
- [ ] Funding proposal draft
- [ ] Demo video
