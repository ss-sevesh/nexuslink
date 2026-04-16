"""WIKI linking pipeline: reads vault notes, finds cross-domain bridges, exports Concept.md files."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import yaml
from loguru import logger

from nexuslink.raw.extraction.entity_extractor import _is_valid_entity
from nexuslink.raw.schemas.models import ExtractedEntity, RawDocument
from nexuslink.wiki.graph.builder import KnowledgeGraph
from nexuslink.wiki.linker.bridge_finder import BridgeFinder
from nexuslink.wiki.linker.embedder import ConceptEmbedder
from nexuslink.wiki.taxonomy.classifier import macro_domain

_WIKI_DIR = Path(__file__).parent.parent          # wiki/
_PAPERS_DIR = _WIKI_DIR / "01-papers"
_CACHE_PATH = _WIKI_DIR / ".cache" / "graph.gpickle"

# Matches [[Some Link]] or [[Some Link|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]")
# Matches "- [[Name]] (entity_type)" with optional inline context "<!-- cx: ... -->"
_ENTITY_LINE_RE = re.compile(
    r"-\s+\[\[([^\]]+)\]\]\s+\(([^)]+)\)"
    r"(?:<!--\s*cx:\s*(.*?)\s*-->)?"
)


async def run_linking(
    threshold: float = 0.65,
    force_rebuild: bool = False,
) -> dict:
    """Read all Paper.md notes from the vault, find cross-domain bridges, export Concept notes.

    Parameters
    ----------
    threshold:
        Minimum cosine similarity to form a bridge.
    force_rebuild:
        Ignore cached graph and rebuild from scratch.

    Returns
    -------
    Summary dict with: papers_processed, total_concepts, total_bridges,
    domains_covered, strongest_bridge, concept_notes_written.
    """
    # ----------------------------------------------------------------
    # 1. Load or initialise knowledge graph
    # ----------------------------------------------------------------
    if not force_rebuild and _CACHE_PATH.exists():
        kg = await asyncio.to_thread(KnowledgeGraph.load, _CACHE_PATH)
        logger.info("Loaded cached knowledge graph")
    else:
        kg = KnowledgeGraph()
        logger.info("Starting fresh knowledge graph")

    # ----------------------------------------------------------------
    # 2. Discover Paper.md notes via obsidiantools + direct scan
    # ----------------------------------------------------------------
    paper_files = await _discover_paper_notes()

    if not paper_files:
        logger.warning("No Paper.md notes found in {}. Run ingestion first.", _PAPERS_DIR)
        return {
            "papers_processed": 0,
            "total_concepts": 0,
            "total_bridges": 0,
            "domains_covered": 0,
            "strongest_bridge": None,
            "concept_notes_written": 0,
        }

    # ----------------------------------------------------------------
    # 3. Parse notes → build entity-by-domain index
    # ----------------------------------------------------------------
    entities_by_domain: dict[str, list[ExtractedEntity]] = {}
    papers_processed = 0

    for note_path in paper_files:
        content = await asyncio.to_thread(note_path.read_text, "utf-8")
        frontmatter, body = _parse_frontmatter(content)

        title: str = frontmatter.get("title", note_path.stem) or note_path.stem
        domains: list[str] = _coerce_list(frontmatter.get("domain", []))
        if not domains:
            domains = ["unknown"]

        doc_id = re.sub(r"\W+", "_", title)[:64]
        entities = _extract_entities_from_body(body, doc_id)

        # Build a synthetic RawDocument for graph insertion
        synthetic_doc = RawDocument(
            id=doc_id,
            title=title,
            authors=_coerce_list(frontmatter.get("authors", [])),
            doi=frontmatter.get("doi") or None,
            arxiv_id=frontmatter.get("arxiv_id") or None,
            full_text="",          # not needed at WIKI layer
            source_path=str(note_path),
            year=frontmatter.get("year") or None,
            domain_tags=domains,
        )
        kg.add_paper(synthetic_doc, entities)

        for domain in domains:
            # Collapse cs.CL/cs.LG/stat.ML etc. → single "cs" bucket so they
            # don't create spurious intra-CS "cross-domain" bridges.
            entities_by_domain.setdefault(macro_domain(domain), []).extend(entities)

        papers_processed += 1
        logger.debug("Indexed {!r} ({} entities, domains={})", title, len(entities), domains)

    # ----------------------------------------------------------------
    # 4. Embed and find cross-domain bridges
    # ----------------------------------------------------------------
    embedder = ConceptEmbedder()
    finder = BridgeFinder(embedder)
    bridges = finder.find_bridges(entities_by_domain, threshold=threshold)

    for bridge in bridges:
        kg.add_bridge(bridge)

    # ----------------------------------------------------------------
    # 5. Export Concept.md notes
    # ----------------------------------------------------------------
    concept_count = await kg.export_for_obsidian()

    # ----------------------------------------------------------------
    # 6. Persist state
    # ----------------------------------------------------------------
    await embedder.save_cache_async()
    await asyncio.to_thread(kg.save)

    # ----------------------------------------------------------------
    # 7. Build and log summary
    # ----------------------------------------------------------------
    stats = {
        "papers_processed": papers_processed,
        "total_concepts": kg.node_count("concept"),
        "total_bridges": len(bridges),
        "domains_covered": len(entities_by_domain),
        "strongest_bridge": bridges[0].model_dump() if bridges else None,
        "concept_notes_written": concept_count,
    }
    logger.info(
        "Linking complete — papers={papers_processed}, concepts={total_concepts}, "
        "bridges={total_bridges}, domains={domains_covered}",
        **stats,
    )
    return stats


# ---------------------------------------------------------------------------
# Vault discovery
# ---------------------------------------------------------------------------

async def _discover_paper_notes() -> list[Path]:
    """Return all .md files under wiki/01-papers/, using obsidiantools for discovery.

    Falls back to a direct glob if obsidiantools raises or returns nothing.
    """
    paths: list[Path] = []

    try:
        import obsidiantools.api as otools  # noqa: PLC0415

        vault = await asyncio.to_thread(_connect_vault)
        index: dict = vault.md_file_index  # {note_name: relative_path}
        for name, rel_path in index.items():
            abs_path = _WIKI_DIR / rel_path
            if "01-papers" in rel_path.parts and abs_path.suffix == ".md":
                paths.append(abs_path)

        if paths:
            logger.debug("obsidiantools discovered {} paper notes", len(paths))
            return paths
    except Exception as exc:  # noqa: BLE001
        logger.debug("obsidiantools unavailable ({}), falling back to glob", exc)

    # Direct glob fallback
    if _PAPERS_DIR.exists():
        paths = list(_PAPERS_DIR.glob("*.md"))
    logger.debug("Glob found {} paper notes", len(paths))
    return paths


def _connect_vault():
    import obsidiantools.api as otools  # noqa: PLC0415

    return otools.Vault(_WIKI_DIR).connect()


# ---------------------------------------------------------------------------
# Frontmatter + wikilink parsing
# ---------------------------------------------------------------------------

def _parse_frontmatter(content: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not match:
        return {}, content
    try:
        fm: dict = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("YAML parse error in frontmatter: {}", exc)
        fm = {}
    return fm, content[match.end():]


def _extract_entities_from_body(body: str, source_doc_id: str) -> list[ExtractedEntity]:
    """Parse the ## Entities section of a Paper.md note into ExtractedEntity objects.

    Expects lines of the form:  - [[Name]] (entity_type)
    Falls back to any [[wikilink]] in the Entities section.
    """
    entities: list[ExtractedEntity] = []
    seen: set[str] = set()

    # Find Entities section.
    # Stop at the next REAL section header (## Capital-word) not at ## inside HTML
    # comments (e.g. "## **3.4 Winograd-Style Tasks**" that PDF extraction embeds).
    section_match = re.search(r"##\s+Entities\s*\n(.*?)(?=\n##\s+[A-Z]|\Z)", body, re.DOTALL)
    if not section_match:
        return entities

    section_text = section_match.group(1)

    # Try structured "- [[name]] (type)" lines first
    # Optionally includes "<!-- cx: sentence -->" for context sentence
    for m in _ENTITY_LINE_RE.finditer(section_text):
        name = m.group(1).strip()
        raw_type = m.group(2).strip().lower()
        entity_type = raw_type if raw_type in _VALID_TYPES else "phenomenon"
        context = m.group(3).strip() if m.group(3) else ""
        if name not in seen and _is_valid_entity(name):
            seen.add(name)
            entities.append(ExtractedEntity(
                name=name,
                entity_type=entity_type,  # type: ignore[arg-type]
                source_doc_id=source_doc_id,
                context_sentence=context,
            ))

    # Fallback: any [[wikilink]] in the section
    if not entities:
        for m in _WIKILINK_RE.finditer(section_text):
            name = m.group(1).strip()
            if name and name not in seen and _is_valid_entity(name):
                seen.add(name)
                entities.append(ExtractedEntity(
                    name=name,
                    entity_type="phenomenon",
                    source_doc_id=source_doc_id,
                    context_sentence="",
                ))

    return entities


_VALID_TYPES = {"chemical", "gene", "method", "material", "phenomenon", "organism"}


def _coerce_list(val: object) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str) and val:
        return [val]
    return []
