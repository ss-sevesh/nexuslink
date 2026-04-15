"""Knowledge graph: NetworkX DiGraph with papers, concepts, and typed cross-domain edges."""

from __future__ import annotations

import asyncio
import pickle
import re
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

from nexuslink.raw.schemas.models import ExtractedEntity, RawDocument
from nexuslink.wiki.linker.bridge_finder import ConceptBridge

_WIKI_DIR = Path(__file__).parent.parent          # wiki/
_CACHE_PATH = _WIKI_DIR / ".cache" / "graph.gpickle"


class KnowledgeGraph:
    """Directed knowledge graph over scientific papers and extracted concepts.

    Node kinds
    ----------
    ``type="paper"``   — one per :class:`RawDocument`
    ``type="concept"`` — one per unique entity name

    Edge kinds
    ----------
    ``relation="mentions"``   — paper → concept
    ``relation=<bridge_type>``— concept ↔ concept (cross-domain bridge)
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_paper(self, doc: RawDocument, entities: list[ExtractedEntity]) -> None:
        """Add a paper node, its concept nodes, and 'mentions' edges."""
        paper_id = f"paper::{doc.id}"
        self._graph.add_node(
            paper_id,
            type="paper",
            title=doc.title,
            authors=doc.authors,
            doi=doc.doi or "",
            arxiv_id=doc.arxiv_id or "",
            year=doc.year,
            domains=doc.domain_tags,
        )

        for entity in entities:
            concept_id = f"concept::{entity.name}"
            if not self._graph.has_node(concept_id):
                self._graph.add_node(
                    concept_id,
                    type="concept",
                    name=entity.name,
                    entity_type=entity.entity_type,
                    domains=doc.domain_tags[:],  # mutable copy per paper
                    context_sentence=entity.context_sentence or "",
                )
            else:
                # Accumulate domains across papers; keep first context sentence
                existing: list[str] = self._graph.nodes[concept_id]["domains"]
                for d in doc.domain_tags:
                    if d not in existing:
                        existing.append(d)
                # Fill in context sentence if missing (concept seen in earlier paper without it)
                if not self._graph.nodes[concept_id].get("context_sentence"):
                    self._graph.nodes[concept_id]["context_sentence"] = entity.context_sentence or ""

            self._graph.add_edge(paper_id, concept_id, relation="mentions")

        logger.debug("Added paper {!r} with {} entities", doc.title, len(entities))

    def add_bridge(self, bridge: ConceptBridge) -> None:
        """Add a bidirectional cross-domain edge for *bridge*."""
        id_a = f"concept::{bridge.entity_a}"
        id_b = f"concept::{bridge.entity_b}"

        # Add concept nodes if they don't exist yet (edge-only import path)
        for nid, domain, etype in (
            (id_a, bridge.domain_a, bridge.entity_type_a),
            (id_b, bridge.domain_b, bridge.entity_type_b),
        ):
            if not self._graph.has_node(nid):
                self._graph.add_node(
                    nid,
                    type="concept",
                    name=nid.split("::", 1)[1],
                    entity_type=etype,
                    domains=[domain],
                )

        attrs: dict[str, Any] = {
            "relation": bridge.bridge_type,
            "similarity": bridge.similarity_score,
            "domain_a": bridge.domain_a,
            "domain_b": bridge.domain_b,
        }
        self._graph.add_edge(id_a, id_b, **attrs)
        self._graph.add_edge(id_b, id_a, **attrs)  # bidirectional

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_cross_domain_clusters(self) -> list[list[str]]:
        """Return connected components of concept nodes that span ≥2 domains.

        Each returned list contains the bare concept *names* (not node IDs).
        """
        undirected = self._graph.to_undirected()
        clusters: list[list[str]] = []

        for component in nx.connected_components(undirected):
            concept_ids = [n for n in component if self._graph.nodes[n].get("type") == "concept"]
            if len(concept_ids) < 2:
                continue
            all_domains: set[str] = set()
            for cid in concept_ids:
                all_domains.update(self._graph.nodes[cid].get("domains", []))
            if len(all_domains) >= 2:
                names = sorted(self._graph.nodes[cid]["name"] for cid in concept_ids)
                clusters.append(names)

        return sorted(clusters, key=len, reverse=True)

    def get_bridges(self) -> list[ConceptBridge]:
        """Reconstruct all ConceptBridge objects from cross-domain edges, deduplicated."""
        _bridge_relations = {"analogous", "enables", "extends", "contradicts"}
        seen: set[frozenset[str]] = set()
        bridges: list[ConceptBridge] = []

        for src, dst, data in self._graph.edges(data=True):
            if data.get("relation") not in _bridge_relations:
                continue
            src_attrs = self._graph.nodes[src]
            dst_attrs = self._graph.nodes[dst]
            if src_attrs.get("type") != "concept" or dst_attrs.get("type") != "concept":
                continue
            pair: frozenset[str] = frozenset([src, dst])
            if pair in seen:
                continue
            seen.add(pair)
            bridges.append(
                ConceptBridge(
                    entity_a=src_attrs.get("name", src),
                    entity_b=dst_attrs.get("name", dst),
                    similarity_score=float(data.get("similarity", 0.0)),
                    domain_a=str(data.get("domain_a", "")),
                    domain_b=str(data.get("domain_b", "")),
                    bridge_type=data["relation"],  # type: ignore[arg-type]
                    entity_type_a=src_attrs.get("entity_type", ""),
                    entity_type_b=dst_attrs.get("entity_type", ""),
                )
            )

        return sorted(bridges, key=lambda b: b.similarity_score, reverse=True)

    def node_count(self, node_type: str | None = None) -> int:
        if node_type is None:
            return self._graph.number_of_nodes()
        return sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == node_type)

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Obsidian export
    # ------------------------------------------------------------------

    async def export_for_obsidian(self) -> int:
        """Write/update a Concept.md file for every concept node.

        Returns the number of files written.
        """
        concepts_dir = _WIKI_DIR / "02-concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for node_id, attrs in self._graph.nodes(data=True):
            if attrs.get("type") != "concept":
                continue

            name: str = attrs.get("name", node_id)
            entity_type: str = attrs.get("entity_type", "phenomenon")
            domains: list[str] = attrs.get("domains", [])
            context_sentence: str = attrs.get("context_sentence", "")

            # Concepts this node bridges to
            bridges: list[tuple[str, str, float]] = []  # (other_name, relation, similarity)
            for _, tgt, edata in self._graph.out_edges(node_id, data=True):
                if self._graph.nodes[tgt].get("type") == "concept":
                    bridges.append((
                        self._graph.nodes[tgt]["name"],
                        edata.get("relation", "related"),
                        edata.get("similarity", 0.0),
                    ))

            # Papers that mention this concept
            papers: list[str] = []
            for src, _, _ in self._graph.in_edges(node_id, data=True):
                if self._graph.nodes[src].get("type") == "paper":
                    papers.append(self._graph.nodes[src].get("title", src))

            content = _render_concept_note(name, entity_type, domains, bridges, papers, context_sentence)
            path = concepts_dir / f"{_sanitize(name)}.md"
            await asyncio.to_thread(path.write_text, content, "utf-8")
            count += 1

        logger.info("Exported {} concept notes to wiki/02-concepts/", count)
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        p = path or _CACHE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self._graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved knowledge graph ({} nodes, {} edges) to {}",
                    self._graph.number_of_nodes(), self._graph.number_of_edges(), p)

    @classmethod
    def load(cls, path: Path | None = None) -> "KnowledgeGraph":
        p = path or _CACHE_PATH
        with open(p, "rb") as fh:
            graph: nx.DiGraph = pickle.load(fh)
        kg = cls()
        kg._graph = graph
        logger.info("Loaded knowledge graph ({} nodes, {} edges) from {}",
                    graph.number_of_nodes(), graph.number_of_edges(), p)
        return kg


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_concept_note(
    name: str,
    entity_type: str,
    domains: list[str],
    bridges: list[tuple[str, str, float]],
    papers: list[str],
    context_sentence: str = "",
) -> str:
    domains_yaml = "[" + ", ".join(f'"{d}"' for d in domains) + "]" if domains else "[]"

    related_lines = "\n".join(
        f"- [[{b_name}]] ({relation}, similarity: {sim:.2f})"
        for b_name, relation, sim in sorted(bridges, key=lambda x: x[2], reverse=True)
    ) or "<!-- none yet -->"

    paper_lines = "\n".join(f"- [[{_sanitize(p)}]]" for p in papers) or "<!-- none yet -->"

    cross_domain = "\n".join(
        f"- [[{b_name}]] → {relation}"
        for b_name, relation, _ in bridges
        if relation in ("analogous", "enables", "extends", "contradicts")
    ) or "<!-- none yet -->"

    return f"""\
---
name: "{_escape(name)}"
domains: {domains_yaml}
type: {entity_type}
tags: []
---

## Definition

{context_sentence}

## Related Concepts

{related_lines}

## Appears In

{paper_lines}

## Cross-Domain Bridges

{cross_domain}

## All References (live)

```dataview
LIST
FROM [[]]
SORT file.folder ASC
```
"""


def _sanitize(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s)
    return s.strip(". ")[:200] or "untitled"


def _escape(s: str) -> str:
    return s.replace('"', '\\"')
