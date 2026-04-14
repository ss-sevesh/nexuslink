# NexusLink — Cross-Domain Research Hypothesis Engine

## What This Project Does
NexusLink ingests research papers from multiple scientific domains, builds a cross-domain knowledge graph, identifies non-obvious conceptual bridges between fields, and generates novel, citation-backed hypothesis reports that suggest unexplored research directions.

## Architecture: 3-Layer Pipeline (Raw → Wiki → LLM)

### Layer 1: RAW (Ingestion & Extraction)
- **ingestion/**: Parses PDFs, ArXiv links, DOIs, and BibTeX imports. Uses `pymupdf4llm` for PDF→Markdown, `arxiv` API client for ArXiv papers.
- **extraction/**: NER for scientific entities (chemicals, genes, methods, materials). Extracts claims, findings, methodology sections. Uses spaCy + domain-specific models.
- **store/**: Stores raw text chunks with metadata. Vector embeddings via `sentence-transformers` for semantic search. ChromaDB for vector store.
- **schemas/**: Pydantic v2 models — `RawDocument`, `ExtractedEntity`, `Citation`, `Claim`, `Methodology`.
- **Output**: Each paper becomes a structured JSON with entities, claims, citations, and embeddings.

### Layer 2: WIKI (Knowledge Graph & Linking)
**The `wiki/` directory is the Obsidian vault.** Open it directly in Obsidian. All notes use `[[wikilinks]]` — never plain Markdown links for internal references. Python scripts write output as `.md` files directly into `wiki/` subdirectories; Obsidian picks them up automatically.

- **linker/**: The core innovation — finds cross-domain concept bridges. Example: "Casimir effect" (physics) ↔ "van der Waals forces" (chemistry) ↔ "gecko adhesion" (biology). Uses embedding similarity + ontology alignment.
- **graph/**: Builds a NetworkX/neo4j knowledge graph. Nodes = concepts/entities, Edges = relations (cites, contradicts, extends, analogous_to, enables). Supports temporal evolution of ideas.
- **taxonomy/**: Maps papers to domain ontologies (e.g., Physics→Quantum→Casimir, Biology→Biomimetics→Adhesion). Uses MESH, ArXiv categories, and custom taxonomies.
- **citations/**: Full academic citation pipeline — BibTeX/CSL-JSON storage, DOI resolution, citation graph analysis. Tracks which claims cite which sources.
- **templates/**: Obsidian templates for each entity type — `Paper.md`, `Concept.md`, `Hypothesis.md`, `Method.md`. Scripts in `raw/ingestion/` and `llm/hypothesis/` render these templates when writing output notes.
- **Output**: A queryable knowledge graph where cross-domain links are first-class citizens, browsable and editable in Obsidian.

### Layer 3: LLM (Hypothesis Generation)
- **hypothesis/**: Takes cross-domain bridges from wiki layer and generates structured hypotheses. Format: "If [finding from domain A] and [mechanism from domain B], then [novel prediction for domain C]". Each hypothesis includes confidence score, required experiments, and potential impact.
- **scoring/**: Ranks hypotheses by: novelty (not already published), feasibility (can be tested), impact (significance if true), cross-domain-span (more domains = more novel).
- **reports/**: Generates publication-ready reports in LaTeX and Markdown. Sections: Abstract, Cross-Domain Analysis, Generated Hypotheses (ranked), Evidence Map, Suggested Experiments, Full Bibliography.
- **prompts/**: Jinja2 prompt templates for each stage — entity extraction, link discovery, hypothesis generation, critique, refinement.
- **validation/**: Checks generated hypotheses against known literature for contradictions, verifies citation accuracy, flags speculative vs. grounded claims.
- **Output**: A ranked hypothesis report with full citations, confidence scores, and experimental roadmaps.

## Tech Stack
- **Runtime**: Python 3.12, managed with `uv`
- **Core**: pydantic v2, networkx, sentence-transformers, chromadb
- **PDF/Papers**: pymupdf4llm, arxiv, habanero (DOI resolution)
- **NLP**: spaCy, sciSpaCy (biomedical NER)
- **Tech Stack**:
- **LLM**: anthropic SDK (Claude API), httpx (Ollama local inference)
- **API**: FastAPI + uvicorn
- **Testing**: pytest, pytest-asyncio
- **Linting**: ruff
- **Citations**: citeproc-py, bibtexparser
- **Vault**: obsidiantools (programmatic read/write of the Obsidian vault in `wiki/`)

## Key Commands

### 🚀 Quickstart Demo
```bash
uv run python demo/run_demo.py
```
Open `demo/demo-vault/` in Obsidian — explore papers, concepts, bridges, and generated hypotheses. 

**LLM Configuration:**
- **Local Inference (Zero-Cost):** Set `OLLAMA_MODEL` (e.g., `llama3.1:8b`) to use a local Ollama instance. The demo will auto-detect a running Ollama server on `http://localhost:11434` if no keys are provided.
- **Cloud Inference:** Set `ANTHROPIC_API_KEY` for Claude API generation.

### Pipeline Commands
- `uv run nexuslink ingest <pdf|arxiv_id|doi>` — Ingest a single paper
- `uv run nexuslink ingest-batch <folder>` — Ingest all PDFs in a folder
- `uv run nexuslink link [--threshold 0.65] [--force-rebuild]` — Run cross-domain linking
- `uv run nexuslink hypothesize [--top-n 5] [--skip-validation]` — Generate hypotheses
- `uv run nexuslink run <source1> <source2> ...` — Full pipeline end-to-end
- `uv run nexuslink status [--vault-path ./wiki/]` — Print vault statistics
- `uv run uvicorn nexuslink.api.app:app --reload` — Start API dev server (http://localhost:8000)
- `uv run pytest tests/` — Run tests
- `uv run ruff check .` — Lint

## Resume Context
- Check PROGRESS.md for current phase
- Each layer is independently testable
- Start with RAW layer, then WIKI, then LLM

## Funding Targets
- NSF Convergence Accelerator (Track: AI4Science)
- DARPA AIE (AI Exploration)
- Wellcome Trust Discovery Award
- Sloan Research Fellowship tools track
