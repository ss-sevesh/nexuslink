"""
NexusLink · End-to-End Demo
============================
Run with:   uv run python demo/run_demo.py
            (from the nexuslink/ project root)

What it does
------------
1.  Creates demo/demo-vault/ (Obsidian-ready) with .obsidian/ config and templates/
2.  Uses 3 hardcoded sample documents (no real PDFs needed)
3.  Runs the entity-extraction layer (spaCy optional — falls back gracefully)
4.  Writes Paper.md notes into the vault
5.  Writes Concept.md notes with [[wikilinks]] back to papers
6.  Runs the embedding + bridge-finder across 3 domains
7.  Builds the knowledge graph
8.  Exports graph notes to demo-vault/
9.  If ANTHROPIC_API_KEY is set → LLM hypothesis + scoring + report
    Otherwise              → writes mock hypothesis from demo/mock_hypothesis.md
10. Prints rich progress throughout
Final message: "Open demo/demo-vault/ in Obsidian to see the results"
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Force UTF-8 stdout/stderr so emoji + rich markup work on Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Ensure the nexuslink package is importable when running as a plain script.
#
# Layout on disk:
#   Nexus-link/          ← _REPO_ROOT  (must be on sys.path for 'import nexuslink')
#     nexuslink/         ← _PROJECT_ROOT  (the Python package directory)
#       demo/
#         run_demo.py    ← __file__
# ---------------------------------------------------------------------------
_DEMO_DIR    = Path(__file__).parent.resolve()   # Nexus-link/nexuslink/demo/
_PROJECT_ROOT = _DEMO_DIR.parent                 # Nexus-link/nexuslink/  (pyproject.toml lives here)
_REPO_ROOT    = _PROJECT_ROOT.parent             # Nexus-link/  (parent of the package dir)

for _p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Rich console  (optional — degrades to plain print if not installed)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.rule import Rule
    from rich.table import Table
    from rich import print as rprint

    # legacy_windows=False uses VT sequences instead of the cp1252 Win32 renderer
    console = Console(legacy_windows=False, highlight=False)
    RICH = True
except ImportError:
    RICH = False

    class _FallbackConsole:  # type: ignore[override]
        def print(self, *args, **kw):
            print(*args)
        def rule(self, title=""):
            print(f"\n{'='*60}")
            if title:
                print(f"  {title}")
            print(f"{'='*60}")
        def log(self, *args, **kw):
            print("[LOG]", *args)

    console = _FallbackConsole()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_WIKI_DIR      = _PROJECT_ROOT / "wiki"
_OBSIDIAN_SRC  = _WIKI_DIR / ".obsidian"
_TEMPLATES_SRC = _WIKI_DIR / "templates"
_DEMO_VAULT    = _DEMO_DIR / "demo-vault"
_MOCK_HYP_SRC  = _DEMO_DIR / "mock_hypothesis.md"


# ===========================================================================
# Step helpers
# ===========================================================================

def step(icon: str, title: str, detail: str = "") -> None:
    if RICH:
        msg = f"[bold cyan]{icon}[/bold cyan]  [bold]{title}[/bold]"
        if detail:
            msg += f"  [dim]{detail}[/dim]"
        console.print(msg)
    else:
        print(f"{icon}  {title}" + (f"  ({detail})" if detail else ""))


def ok(msg: str) -> None:
    if RICH:
        console.print(f"   [green]OK[/green] {msg}")
    else:
        print(f"   OK  {msg}")


def warn(msg: str) -> None:
    if RICH:
        console.print(f"   [yellow]WARN[/yellow]  {msg}")
    else:
        print(f"   WARN  {msg}")


def err(msg: str) -> None:
    if RICH:
        console.print(f"   [red]ERR[/red]  {msg}")
    else:
        print(f"   ERR  {msg}")


# ===========================================================================
# 1. Sample documents (no PDFs needed)
# ===========================================================================

_SAMPLE_PAPERS = [
    {
        "title": "CRISPR-Cas9 Adaptive Immunity in Bacteria",
        "authors": ["J. A. Doudna", "E. Charpentier", "M. Jinek"],
        "doi": "10.1126/science.demo-001",
        "year": 2012,
        "domain": "molecular_biology",
        "abstract": (
            "The CRISPR-Cas9 system provides adaptive immunity in bacteria against "
            "bacteriophage by using guide RNA to direct the Cas9 endonuclease to "
            "complementary DNA sequences, inducing double-strand breaks and triggering "
            "homology-directed repair."
        ),
        "full_text": (
            "Clustered regularly interspaced short palindromic repeats (CRISPR) and the "
            "associated Cas9 protein form a two-component system for targeted genome editing. "
            "A guide RNA (gRNA) directs Cas9 to a specific genomic locus via Watson-Crick "
            "base pairing, whereupon Cas9 cleaves both strands of DNA, producing a "
            "double-strand break (DSB). The cell may repair this break via non-homologous "
            "end joining (NHEJ) or, when a repair template is provided, via homology-directed "
            "repair (HDR). In bacteria, CRISPR arrays encode spacer sequences derived from "
            "previous bacteriophage infections, providing immunological memory against future "
            "phage challenge. This mechanism represents a form of adaptive immunity at the "
            "molecular level, where the genome itself carries a heritable record of past "
            "environmental threats."
        ),
        "entities": [
            ("CRISPR-Cas9",           "method",      "CRISPR-Cas9 system provides adaptive immunity in bacteria."),
            ("guide RNA",             "method",      "A guide RNA (gRNA) directs Cas9 to a specific genomic locus."),
            ("double-strand break",   "phenomenon",  "Cas9 cleaves both strands of DNA, producing a double-strand break."),
            ("homology-directed repair","method",    "When a repair template is provided, repair occurs via homology-directed repair."),
            ("bacteriophage",         "organism",    "CRISPR arrays encode spacers from bacteriophage infections."),
        ],
    },
    {
        "title": "Self-Healing Polymers via Dynamic Covalent Chemistry",
        "authors": ["B. C. Tee", "C. Wang", "R. Allen", "Z. Bao"],
        "doi": "10.1021/demo-002",
        "year": 2020,
        "domain": "materials_science",
        "abstract": (
            "Self-healing polymers employing dynamic covalent bonds based on the Diels-Alder "
            "reaction autonomously repair crack propagation damage without external agents, "
            "recovering tensile strength through autonomous repair mechanisms."
        ),
        "full_text": (
            "Self-healing polymers represent a new class of smart materials capable of autonomously "
            "repairing mechanical damage. The most successful systems exploit dynamic covalent bonds "
            "— particularly Diels-Alder (DA) reaction adducts between furan and maleimide groups — "
            "that can reversibly break and reform under thermal stimulus. When a crack propagates "
            "through the polymer network, DA adducts at the crack front undergo retro-Diels-Alder "
            "dissociation, freeing reactive chain ends. Upon heating to 60–80 °C, these ends diffuse "
            "across the crack interface and reform DA bonds, restoring the covalent network. This "
            "autonomous repair process requires no external catalyst, adhesive, or encapsulated "
            "healing agent. Tensile strength recovery exceeds 95% after a single heal cycle. The "
            "dynamic covalent bond approach thus enables materials with crack propagation resistance "
            "far exceeding conventional thermosets."
        ),
        "entities": [
            ("self-healing polymer",   "material",   "Self-healing polymers are capable of autonomously repairing mechanical damage."),
            ("dynamic covalent bond",  "phenomenon", "Dynamic covalent bonds reversibly break and reform under thermal stimulus."),
            ("Diels-Alder reaction",   "method",     "Diels-Alder reaction adducts between furan and maleimide groups are key to self-healing."),
            ("crack propagation",      "phenomenon", "When a crack propagates through the network, DA adducts undergo retro-DA dissociation."),
            ("autonomous repair",      "phenomenon", "This autonomous repair process requires no external catalyst or adhesive."),
        ],
    },
    {
        "title": "Error-Correcting Codes in Distributed Systems",
        "authors": ["L. Lamport", "R. Shostak", "M. Pease"],
        "doi": "10.1145/demo-003",
        "year": 2019,
        "domain": "computer_science",
        "abstract": (
            "Error-correcting codes with redundancy enable fault tolerance in distributed "
            "systems against Byzantine faults. Self-repair mechanisms allow nodes to detect "
            "and correct corrupted state autonomously."
        ),
        "full_text": (
            "Error-correcting codes (ECC) are mathematical constructs that introduce controlled "
            "redundancy into transmitted or stored data so that errors can be detected and "
            "corrected without retransmission. In distributed systems, ECC principles underpin "
            "Byzantine fault tolerance (BFT): a system of n nodes can tolerate f Byzantine "
            "faults — nodes that behave arbitrarily or maliciously — provided n ≥ 3f + 1. "
            "The redundancy requirement ensures a quorum of n − f honest nodes always "
            "outnumbers the f faulty nodes. Self-repair in distributed systems refers to "
            "the ability of nodes to autonomously detect inconsistency (via checksums, Merkle "
            "trees, or consensus voting) and restore consistent state from surviving correct "
            "replicas. Unlike physical self-healing, distributed self-repair is an informational "
            "process: the system corrects a corrupted data replica by comparing it against a "
            "majority of uncorrupted replicas and overwriting with the correct value. Fault "
            "tolerance in this context is a design property that must be explicitly engineered "
            "using redundancy and consensus protocols."
        ),
        "entities": [
            ("error-correcting code", "method",     "Error-correcting codes introduce controlled redundancy into data."),
            ("redundancy",            "phenomenon", "Redundancy ensures a quorum of honest nodes outnumbers faulty ones."),
            ("fault tolerance",       "phenomenon", "Byzantine fault tolerance: system tolerates f Byzantine faults."),
            ("self-repair",           "phenomenon", "Self-repair: nodes autonomously detect inconsistency and restore state."),
            ("Byzantine fault",       "phenomenon", "Byzantine faults are nodes that behave arbitrarily or maliciously."),
        ],
    },
]


# ===========================================================================
# 2. Vault setup
# ===========================================================================

async def setup_vault() -> None:
    """Create demo-vault/ with .obsidian config and templates."""
    step("[DIR]", "Setting up demo vault", str(_DEMO_VAULT))
    try:
        _DEMO_VAULT.mkdir(parents=True, exist_ok=True)

        # Sub-directories expected by graph builder & export
        for sub in ("01-papers", "02-concepts", "03-hypotheses", "04-reports", "templates"):
            (_DEMO_VAULT / sub).mkdir(exist_ok=True)

        # Copy .obsidian config if it exists
        vault_obsidian = _DEMO_VAULT / ".obsidian"
        if _OBSIDIAN_SRC.exists() and not vault_obsidian.exists():
            await asyncio.to_thread(shutil.copytree, _OBSIDIAN_SRC, vault_obsidian)
            ok("Copied .obsidian/ config")
        elif vault_obsidian.exists():
            ok(".obsidian/ config already present")
        else:
            vault_obsidian.mkdir(exist_ok=True)
            warn(".obsidian/ source not found — created empty directory")

        # Copy templates
        vault_templates = _DEMO_VAULT / "templates"
        if _TEMPLATES_SRC.exists():
            for tpl in _TEMPLATES_SRC.glob("*.md"):
                dest = vault_templates / tpl.name
                if not dest.exists():
                    await asyncio.to_thread(shutil.copy2, tpl, dest)
            ok("Copied Obsidian templates")
        else:
            warn("wiki/templates/ not found — skipping template copy")

        ok(f"Vault ready at {_DEMO_VAULT}")
    except Exception as exc:
        err(f"Vault setup error: {exc}")
        raise


# ===========================================================================
# 3. Build RawDocument objects from hardcoded data
# ===========================================================================

def build_documents():
    """Return list[RawDocument] from the hardcoded sample data."""
    step("[DOC]", "Building sample documents", "3 papers - biology / materials / CS")
    try:
        from nexuslink.raw.schemas.models import RawDocument
        docs = []
        for p in _SAMPLE_PAPERS:
            doc = RawDocument(
                title=p["title"],
                authors=p["authors"],
                doi=p["doi"],
                full_text=p["full_text"],
                source_path=f"demo/{p['domain']}/sample.txt",
                year=p["year"],
                abstract=p["abstract"],
                domain_tags=[p["domain"]],
            )
            docs.append(doc)
            ok(f"  {doc.title[:55]}…")
        return docs
    except Exception as exc:
        err(f"Document build failed: {exc}")
        raise


# ===========================================================================
# 4. Entity extraction (use hardcoded entities; spaCy optional)
# ===========================================================================

def extract_entities(docs):
    """Return dict[doc_id, list[ExtractedEntity]] using hardcoded entity lists."""
    step("[NER]", "Extracting entities", "using hardcoded entities (no spaCy model required)")
    try:
        from nexuslink.raw.schemas.models import ExtractedEntity

        result: dict[str, list] = {}
        for doc, paper_data in zip(docs, _SAMPLE_PAPERS):
            entities = []
            for name, etype, ctx in paper_data["entities"]:
                entities.append(
                    ExtractedEntity(
                        name=name,
                        entity_type=etype,          # type: ignore[arg-type]
                        source_doc_id=doc.id,
                        context_sentence=ctx,
                    )
                )
            result[doc.id] = entities
            ok(f"  [{paper_data['domain']}] {len(entities)} entities extracted")
        return result
    except Exception as exc:
        err(f"Entity extraction failed: {exc}")
        raise


# ===========================================================================
# 5. Write Paper.md notes
# ===========================================================================

def _safe_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip(". ")[:200] or "untitled"


async def write_paper_notes(docs, entities_by_doc: dict) -> None:
    """Write one Paper.md per document into demo-vault/papers/."""
    step("[NOTE]", "Writing Paper.md notes", "demo-vault/papers/")
    papers_dir = _DEMO_VAULT / "01-papers"
    papers_dir.mkdir(exist_ok=True)

    try:
        for doc, paper_data in zip(docs, _SAMPLE_PAPERS):
            entities = entities_by_doc.get(doc.id, [])
            entity_links = "\n".join(f"- [[{e.name}]]" for e in entities)

            authors_yaml = "[" + ", ".join(f'"{a}"' for a in doc.authors) + "]"
            domain_tags  = "[" + ", ".join(f'"{t}"' for t in (doc.domain_tags or ["unclassified"])) + "]"

            content = f"""\
---
title: "{doc.title}"
authors: {authors_yaml}
doi: "{doc.doi or ''}"
domain: "{paper_data['domain']}"
year: {doc.year or ''}
tags: {domain_tags}
---

## Summary

{doc.abstract}

## Key Findings

- Demonstrates {entities[0].name if entities else 'key mechanism'} as a core process.
- Provides evidence for cross-domain analogies with repair and fault-tolerance systems.

## Methods

{", ".join(f"[[{e.name}]]" for e in entities if e.entity_type == "method") or "See full text."}

## Entities

{entity_links}

<!-- entities linked as wikilinks above -->

## References

- {doc.doi or 'No DOI assigned'}
"""
            filename = _safe_filename(doc.title) + ".md"
            path = papers_dir / filename
            await asyncio.to_thread(path.write_text, content, "utf-8")
            ok(f"  Wrote {filename}")
    except Exception as exc:
        err(f"Paper note writing failed: {exc}")
        raise


# ===========================================================================
# 6. Write Concept.md notes
# ===========================================================================

async def write_concept_notes(docs, entities_by_doc: dict) -> None:
    """Write one Concept.md per unique entity, with [[wikilinks]] back to papers."""
    step("[CON]", "Writing Concept.md notes", "demo-vault/concepts/")
    concepts_dir = _DEMO_VAULT / "02-concepts"
    concepts_dir.mkdir(exist_ok=True)

    # Aggregate: entity_name → {type, papers, domains, context}
    concept_map: dict[str, dict] = {}
    try:
        for doc, paper_data in zip(docs, _SAMPLE_PAPERS):
            for entity in entities_by_doc.get(doc.id, []):
                if entity.name not in concept_map:
                    concept_map[entity.name] = {
                        "type":    entity.entity_type,
                        "papers":  [],
                        "domains": [],
                        "context": entity.context_sentence,
                    }
                concept_map[entity.name]["papers"].append(doc.title)
                if paper_data["domain"] not in concept_map[entity.name]["domains"]:
                    concept_map[entity.name]["domains"].append(paper_data["domain"])

        for name, info in concept_map.items():
            paper_links = "\n".join(f"- [[{p}]]" for p in info["papers"])
            domains_yaml = "[" + ", ".join(f'"{d}"' for d in info["domains"]) + "]"

            content = f"""\
---
name: "{name}"
domains: {domains_yaml}
type: {info["type"]}
tags: []
---

## Definition

{info["context"]}

## Related Concepts

<!-- [[Concept Name]] — brief relation note -->

## Appears In

{paper_links}

## Cross-Domain Bridges

<!-- Populated automatically by the bridge finder -->
"""
            filename = _safe_filename(name) + ".md"
            path = concepts_dir / filename
            await asyncio.to_thread(path.write_text, content, "utf-8")

        ok(f"  Wrote {len(concept_map)} concept notes")
    except Exception as exc:
        err(f"Concept note writing failed: {exc}")
        raise


# ===========================================================================
# 7. Domain classification (lightweight — just return hardcoded tags)
# ===========================================================================

def classify_domains(docs) -> dict[str, str]:
    """Return doc_id → domain string."""
    step("[CLS]", "Classifying domains")
    result = {}
    for doc, paper_data in zip(docs, _SAMPLE_PAPERS):
        result[doc.id] = paper_data["domain"]
        ok(f"  {doc.title[:45]}… → {paper_data['domain']}")
    return result


# ===========================================================================
# 8. Embeddings + bridge finder
# ===========================================================================

async def run_bridge_finder(docs, entities_by_doc: dict, domain_by_doc: dict):
    """Embed entities and find cross-domain bridges."""
    step("[EMB]", "Running embedding + bridge finder")
    try:
        from nexuslink.wiki.linker.embedder import ConceptEmbedder
        from nexuslink.wiki.linker.bridge_finder import BridgeFinder

        embedder = ConceptEmbedder()
        finder   = BridgeFinder(embedder)

        # Group entities by domain
        entities_by_domain: dict[str, list] = {}
        for doc in docs:
            domain = domain_by_doc[doc.id]
            entities_by_domain.setdefault(domain, [])
            entities_by_domain[domain].extend(entities_by_doc.get(doc.id, []))

        ok(f"  Embedding {sum(len(v) for v in entities_by_domain.values())} entities across "
           f"{len(entities_by_domain)} domains…")

        bridges = await asyncio.to_thread(
            finder.find_bridges, entities_by_domain, 0.35  # lower threshold for demo
        )

        await asyncio.to_thread(embedder.save_cache)
        ok(f"  Found {len(bridges)} cross-domain bridges")

        if RICH and bridges:
            tbl = Table(title="Top Bridges", show_header=True, header_style="bold magenta")
            tbl.add_column("Concept A", style="cyan")
            tbl.add_column("Concept B", style="green")
            tbl.add_column("Type",      style="yellow")
            tbl.add_column("Sim",       justify="right")
            for b in bridges[:8]:
                tbl.add_row(b.entity_a, b.entity_b, b.bridge_type, f"{b.similarity_score:.3f}")
            console.print(tbl)
        elif bridges:
            print("   Top bridges:")
            for b in bridges[:5]:
                print(f"     {b.entity_a} ↔ {b.entity_b} ({b.bridge_type}, {b.similarity_score:.3f})")

        return bridges
    except ImportError as exc:
        warn(f"sentence-transformers not available ({exc}); generating mock bridges")
        return _mock_bridges()
    except Exception as exc:
        warn(f"Bridge finder error: {exc}; falling back to mock bridges")
        return _mock_bridges()


def _mock_bridges():
    """Return hard-coded bridges as fallback when sentence-transformers is unavailable."""
    from nexuslink.wiki.linker.bridge_finder import ConceptBridge

    return [
        ConceptBridge(entity_a="CRISPR-Cas9",           entity_b="error-correcting code",
                      similarity_score=0.82, domain_a="molecular_biology", domain_b="computer_science",
                      bridge_type="analogous", entity_type_a="method",    entity_type_b="method"),
        ConceptBridge(entity_a="homology-directed repair", entity_b="redundancy",
                      similarity_score=0.79, domain_a="molecular_biology", domain_b="computer_science",
                      bridge_type="analogous", entity_type_a="method",    entity_type_b="phenomenon"),
        ConceptBridge(entity_a="dynamic covalent bond",  entity_b="self-repair",
                      similarity_score=0.88, domain_a="materials_science", domain_b="computer_science",
                      bridge_type="analogous", entity_type_a="phenomenon", entity_type_b="phenomenon"),
        ConceptBridge(entity_a="Diels-Alder reaction",   entity_b="fault tolerance",
                      similarity_score=0.71, domain_a="materials_science", domain_b="computer_science",
                      bridge_type="enables",  entity_type_a="method",    entity_type_b="phenomenon"),
        ConceptBridge(entity_a="bacteriophage",          entity_b="Byzantine fault",
                      similarity_score=0.67, domain_a="molecular_biology", domain_b="computer_science",
                      bridge_type="analogous", entity_type_a="organism",   entity_type_b="phenomenon"),
        ConceptBridge(entity_a="double-strand break",    entity_b="crack propagation",
                      similarity_score=0.74, domain_a="molecular_biology", domain_b="materials_science",
                      bridge_type="analogous", entity_type_a="phenomenon", entity_type_b="phenomenon"),
        ConceptBridge(entity_a="guide RNA",              entity_b="autonomous repair",
                      similarity_score=0.69, domain_a="molecular_biology", domain_b="materials_science",
                      bridge_type="enables",  entity_type_a="method",    entity_type_b="phenomenon"),
    ]


# ===========================================================================
# 9. Knowledge graph
# ===========================================================================

async def build_knowledge_graph(docs, entities_by_doc: dict, bridges):
    """Build and return a KnowledgeGraph; export notes to demo-vault/."""
    step("[GRF]", "Building knowledge graph")
    try:
        from nexuslink.wiki.graph.builder import KnowledgeGraph

        kg = KnowledgeGraph()
        for doc in docs:
            kg.add_paper(doc, entities_by_doc.get(doc.id, []))
        for bridge in bridges:
            kg.add_bridge(bridge)

        ok(f"  Graph: {kg.node_count('paper')} paper nodes, "
           f"{kg.node_count('concept')} concept nodes, "
           f"{kg.edge_count()} edges")

        # Export concept notes — builder writes to wiki/02-concepts/ by default;
        # we monkey-patch the path to point at demo-vault/concepts/ instead.
        import nexuslink.wiki.graph.builder as _builder_mod
        original_wiki_dir = _builder_mod._WIKI_DIR
        _builder_mod._WIKI_DIR = _DEMO_VAULT
        try:
            n = await kg.export_for_obsidian()
            ok(f"  Exported {n} enriched concept notes to demo-vault/concepts/")
        finally:
            _builder_mod._WIKI_DIR = original_wiki_dir

        return kg
    except Exception as exc:
        err(f"Knowledge graph build failed: {exc}")
        warn("Continuing without graph persistence…")
        return None


# ===========================================================================
# 10a. LLM hypothesis generation (requires ANTHROPIC_API_KEY)
# ===========================================================================

async def run_llm_pipeline(kg, bridges) -> bool:
    """Try to run the real LLM pipeline. Returns True on success."""
    import urllib.request
    
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    ollama_model = os.environ.get("OLLAMA_MODEL", "")
    
    # Auto-detect local Ollama if no keys provided
    if not api_key and not ollama_model:
        try:
            req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.0)
            if req.status == 200:
                ollama_model = "llama3.1:8b"
                os.environ["OLLAMA_MODEL"] = ollama_model
        except Exception:
            pass

    if not api_key and not ollama_model:
        return False

    if ollama_model:
        step("[LLM]", "Running LLM hypothesis generation", f"Ollama local inference ({ollama_model}) detected")
    else:
        step("[LLM]", "Running LLM hypothesis generation", "ANTHROPIC_API_KEY detected")
    try:
        import nexuslink.llm.hypothesis.generator as _gen_mod
        from nexuslink.llm.hypothesis.generator import HypothesisGenerator
        from nexuslink.llm.scoring.ranker import HypothesisRanker
        from nexuslink.llm.reports.writer import ReportWriter
        from nexuslink.wiki.citations.manager import CitationManager

        # Redirect all LLM output to demo-vault so every note lives in one vault
        _orig_hyp_dir = _gen_mod._HYPOTHESES_DIR
        _demo_hyp_dir = _DEMO_VAULT / "03-hypotheses"
        _demo_hyp_dir.mkdir(parents=True, exist_ok=True)
        _gen_mod._HYPOTHESES_DIR = _demo_hyp_dir

        try:
            generator = HypothesisGenerator()
            ranker    = HypothesisRanker()
            writer    = ReportWriter()
            citation_manager = CitationManager()

            hypotheses = await generator.generate(bridges[:5], kg)
            ok(f"  Generated {len(hypotheses)} hypotheses")

            scored = await ranker.rank_all(hypotheses)
            ok(f"  Ranked {len(scored)} hypotheses by novelty × feasibility × impact")

            report_path = await writer.generate_report(scored, kg, citation_manager, vault_path=_DEMO_VAULT)
            ok(f"  Report written → {report_path}")
        finally:
            _gen_mod._HYPOTHESES_DIR = _orig_hyp_dir

        return True
    except Exception as exc:
        warn(f"LLM pipeline failed ({exc}); falling back to mock hypothesis")
        return False


# ===========================================================================
# 10b. Mock hypothesis (no API key needed)
# ===========================================================================

async def write_mock_hypothesis(bridges) -> None:
    """Copy pre-written mock hypothesis into demo-vault/."""
    step("[HYP]", "Writing mock hypothesis", "(no API key - showing template output)")

    hyp_dir  = _DEMO_VAULT / "03-hypotheses"
    rep_dir  = _DEMO_VAULT / "04-reports"
    hyp_dir.mkdir(exist_ok=True)
    rep_dir.mkdir(exist_ok=True)

    # ---- Hypothesis.md -------------------------------------------------------
    if _MOCK_HYP_SRC.exists():
        dest = hyp_dir / "Hypothesis-Self-Repairing-Genetic-Circuits.md"
        await asyncio.to_thread(shutil.copy2, _MOCK_HYP_SRC, dest)
        ok(f"  Hypothesis note → {dest.name}")
    else:
        warn("mock_hypothesis.md not found; writing inline fallback")
        _write_inline_hypothesis(hyp_dir)

    # ---- report.md -----------------------------------------------------------
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bridge_rows = "\n".join(
        f"| [[{b.entity_a}]] | [[{b.entity_b}]] | {b.bridge_type} | {b.similarity_score:.3f} |"
        for b in bridges[:10]
    )

    report_md = f"""\
---
generated_at: "{now}"
papers_analysed: 3
domains: ["molecular_biology", "materials_science", "computer_science"]
bridges_found: {len(bridges)}
hypotheses_generated: 1
status: mock
---

# NexusLink Research Report

> **Note:** This is a demonstration report generated without an LLM API key.
> Set `ANTHROPIC_API_KEY` and re-run to get AI-generated hypotheses.

## Cross-Domain Bridges Detected

| Concept A | Concept B | Type | Similarity |
|-----------|-----------|------|------------|
{bridge_rows}

## Generated Hypotheses

### 1. [[Hypothesis-Self-Repairing-Genetic-Circuits]]

**Confidence:** 0.78 · **Novelty:** 0.91 · **Feasibility:** 0.65

> If CRISPR-Cas9 can autonomously repair DNA double-strand breaks using template-guided
> mechanisms, and self-healing polymers can autonomously repair structural damage using
> dynamic covalent bonds, then biological genetic circuits could be engineered with
> error-correcting redundancy inspired by distributed systems fault tolerance — creating
> self-repairing synthetic gene networks that detect and fix mutations without external
> intervention.

**Spans domains:** [[molecular_biology]] · [[materials_science]] · [[computer_science]]

**Key bridges used:**
- [[CRISPR-Cas9]] ↔ [[error-correcting code]] (analogous, sim=0.82)
- [[dynamic covalent bond]] ↔ [[self-repair]] (analogous, sim=0.88)
- [[bacteriophage]] ↔ [[Byzantine fault]] (analogous, sim=0.67)

See [[Hypothesis-Self-Repairing-Genetic-Circuits]] for full details, suggested experiments,
and evidence mapping.

## Source Papers

- [[CRISPR-Cas9 Adaptive Immunity in Bacteria]]
- [[Self-Healing Polymers via Dynamic Covalent Chemistry]]
- [[Error-Correcting Codes in Distributed Systems]]

## Next Steps

1. Set `ANTHROPIC_API_KEY` and re-run `uv run python demo/run_demo.py` for AI hypotheses
2. Open `demo/demo-vault/` in Obsidian and explore the graph view
3. Add your own papers via `uv run nexuslink ingest <path|arxiv_id|doi>`
"""
    rep_path = rep_dir / "report.md"
    await asyncio.to_thread(rep_path.write_text, report_md, "utf-8")
    ok(f"  Report written → {rep_path.name}")


def _write_inline_hypothesis(hyp_dir: Path) -> None:
    content = """\
---
id: hyp-001-demo
confidence: 0.78
novelty_score: 0.91
domains_spanned: ["molecular_biology", "materials_science", "computer_science"]
status: mock
---

## Hypothesis Statement

If CRISPR-Cas9 can autonomously repair DNA double-strand breaks using template-guided mechanisms,
and self-healing polymers can autonomously repair structural damage using dynamic covalent bonds,
then biological genetic circuits could be engineered with error-correcting redundancy inspired by
distributed systems fault tolerance.

## Cross-Domain Bridge

- [[CRISPR-Cas9]] ↔ [[error-correcting code]] — analogous
- [[dynamic covalent bond]] ↔ [[self-repair]] — extends
- [[bacteriophage]] ↔ [[Byzantine fault]] — analogous

## Evidence From

- [[CRISPR-Cas9 Adaptive Immunity in Bacteria]]
- [[Self-Healing Polymers via Dynamic Covalent Chemistry]]
- [[Error-Correcting Codes in Distributed Systems]]
"""
    (hyp_dir / "Hypothesis-Self-Repairing-Genetic-Circuits.md").write_text(content, "utf-8")


# ===========================================================================
# 11. Final summary
# ===========================================================================

def print_summary(bridges, kg) -> None:
    if RICH:
        console.print()
        console.rule("[bold green]Demo Complete")
        console.print()
        panel_text = (
            "[bold]What was created in [cyan]demo/demo-vault/[/cyan]:[/bold]\n\n"
            "  [cyan]papers/[/cyan]        3 x Paper.md notes with YAML frontmatter\n"
            "  [cyan]concepts/[/cyan]      15 x Concept.md notes with wikilinks\n"
            f"  bridges            {len(bridges)} cross-domain concept bridges found\n"
            "  [cyan]hypotheses/[/cyan]    1 x Hypothesis.md  (mock - full LLM available)\n"
            "  [cyan]reports/[/cyan]       report.md with bridge table & hypothesis summary\n"
            "  [cyan].obsidian/[/cyan]     Ready-to-open Obsidian vault config\n\n"
            "[bold yellow]>> Open demo/demo-vault/ as a vault in Obsidian to see the magic[/bold yellow]\n"
            "   Switch to [bold]Graph View[/bold] to visualise cross-domain bridges.\n\n"
            "[dim]For AI hypotheses: set ANTHROPIC_API_KEY and re-run[/dim]"
        )
        console.print(Panel(panel_text, title="[bold green]NexusLink Demo", border_style="green"))
    else:
        print("\n" + "="*60)
        print("  Demo Complete!")
        print("="*60)
        print("\nCreated in demo/demo-vault/:")
        print("  papers/     -> 3 Paper.md notes")
        print("  concepts/   -> 15 Concept.md notes with wikilinks")
        print(f"  bridges     -> {len(bridges)} cross-domain bridges found")
        print("  hypotheses/ -> Hypothesis.md")
        print("  reports/    -> report.md")
        print("\n>>  Open demo/demo-vault/ as a vault in Obsidian to see the magic")
        print("   (Switch to Graph View to visualise cross-domain bridges)")
        print("\nFor AI hypotheses: set ANTHROPIC_API_KEY and re-run")
        print("=" * 60)


# ===========================================================================
# Main
# ===========================================================================

async def main() -> None:
    if RICH:
        console.print()
        console.print(Panel(
            "[bold cyan]NexusLink[/bold cyan]  Cross-Domain Research Hypothesis Engine\n"
            "[dim]End-to-end demo | 3 papers | biology - materials - CS[/dim]",
            border_style="cyan",
        ))
        console.print()
    else:
        print("\n" + "=" * 60)
        print("  NexusLink -- Cross-Domain Research Hypothesis Engine")
        print("  End-to-end demo | 3 papers | biology / materials / CS")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 1 · Vault Setup")
    await setup_vault()

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 2 · Sample Documents")
    docs = build_documents()

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 3 · Entity Extraction")
    entities_by_doc = extract_entities(docs)

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 4 · Write Paper Notes")
    await write_paper_notes(docs, entities_by_doc)

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 5 · Write Concept Notes")
    await write_concept_notes(docs, entities_by_doc)

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 6 · Domain Classification")
    domain_by_doc = classify_domains(docs)

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 7 · Embeddings + Bridge Finder")
    bridges = await run_bridge_finder(docs, entities_by_doc, domain_by_doc)

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 8 · Knowledge Graph")
    kg = await build_knowledge_graph(docs, entities_by_doc, bridges)

    # ------------------------------------------------------------------ #
    if RICH:
        console.rule("[bold blue]Step 9 · Hypothesis Generation")
    llm_succeeded = await run_llm_pipeline(kg, bridges)
    if not llm_succeeded:
        await write_mock_hypothesis(bridges)

    # ------------------------------------------------------------------ #
    print_summary(bridges, kg)


if __name__ == "__main__":
    asyncio.run(main())
