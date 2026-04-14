"""Graph routes: linking pipeline, bridge queries, knowledge-graph export."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from nexuslink.api.deps import get_nexuslink
from nexuslink.main import NexusLink

router = APIRouter(tags=["graph"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class LinkResult(BaseModel):
    bridges_found: int
    new_concepts: int
    domains_covered: int
    papers_processed: int
    concept_notes_written: int


class BridgeOut(BaseModel):
    entity_a: str
    entity_b: str
    similarity_score: float
    domain_a: str
    domain_b: str
    bridge_type: str
    entity_type_a: str = ""
    entity_type_b: str = ""


class VaultStats(BaseModel):
    total_papers: int
    total_concepts: int
    total_bridges: int
    total_hypotheses: int
    domains_covered: list[str]
    total_vault_notes: int


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    domains: list[str] = Field(default_factory=list)
    entity_type: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    similarity: float | None = None


class GraphExport(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_count: int
    edge_count: int


# ---------------------------------------------------------------------------
# POST /api/link
# ---------------------------------------------------------------------------

@router.post("/link", response_model=LinkResult)
async def run_linking(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
    threshold: float = Query(default=None, ge=0.0, le=1.0),
    force_rebuild: bool = False,
) -> LinkResult:
    """Trigger the cross-domain linking pipeline over all ingested papers."""
    from nexuslink.wiki.linker.pipeline import run_linking as _link  # noqa: PLC0415

    eff_threshold = threshold if threshold is not None else nx._config.similarity_threshold
    logger.info("POST /api/link — threshold={}, force_rebuild={}", eff_threshold, force_rebuild)

    stats = await _link(threshold=eff_threshold, force_rebuild=force_rebuild)

    return LinkResult(
        bridges_found=stats.get("total_bridges", 0),
        new_concepts=stats.get("total_concepts", 0),
        domains_covered=stats.get("domains_covered", 0),
        papers_processed=stats.get("papers_processed", 0),
        concept_notes_written=stats.get("concept_notes_written", 0),
    )


# ---------------------------------------------------------------------------
# GET /api/bridges
# ---------------------------------------------------------------------------

@router.get("/bridges", response_model=list[BridgeOut])
async def list_bridges(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
    domain_a: str | None = Query(default=None),
    domain_b: str | None = Query(default=None),
) -> list[BridgeOut]:
    """Return all cross-domain bridges, sorted by similarity.

    Optionally filter with ``?domain_a=physics&domain_b=biology``.
    """
    kg = _load_graph(nx)
    bridges = kg.get_bridges()

    if domain_a:
        bridges = [b for b in bridges if b.domain_a == domain_a or b.domain_b == domain_a]
    if domain_b:
        bridges = [b for b in bridges if b.domain_a == domain_b or b.domain_b == domain_b]

    return [
        BridgeOut(
            entity_a=b.entity_a,
            entity_b=b.entity_b,
            similarity_score=b.similarity_score,
            domain_a=b.domain_a,
            domain_b=b.domain_b,
            bridge_type=b.bridge_type,
            entity_type_a=b.entity_type_a,
            entity_type_b=b.entity_type_b,
        )
        for b in bridges
    ]


# ---------------------------------------------------------------------------
# GET /api/graph/stats
# ---------------------------------------------------------------------------

@router.get("/graph/stats", response_model=VaultStats)
async def graph_stats(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
) -> VaultStats:
    """Return full vault statistics (paper count, concept count, bridges, domains)."""
    stats = await nx.status()
    return VaultStats(**stats)


# ---------------------------------------------------------------------------
# GET /api/graph/export
# ---------------------------------------------------------------------------

@router.get("/graph/export", response_model=GraphExport)
async def export_graph(
    nx: Annotated[NexusLink, Depends(get_nexuslink)],
) -> GraphExport:
    """Export the full knowledge graph as JSON nodes + edges for visualisation."""
    kg = _load_graph(nx)
    g = kg._graph

    nodes: list[GraphNode] = []
    for node_id, attrs in g.nodes(data=True):
        nodes.append(GraphNode(
            id=node_id,
            type=attrs.get("type", "unknown"),
            label=attrs.get("title") or attrs.get("name") or node_id,
            domains=attrs.get("domains") or [],
            entity_type=attrs.get("entity_type") or "",
        ))

    edges: list[GraphEdge] = []
    for src, dst, data in g.edges(data=True):
        edges.append(GraphEdge(
            source=src,
            target=dst,
            relation=data.get("relation", ""),
            similarity=data.get("similarity"),
        ))

    return GraphExport(
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_graph(nx: NexusLink):
    from nexuslink.wiki.graph.builder import KnowledgeGraph  # noqa: PLC0415

    cache = nx._vault / ".cache" / "graph.gpickle"
    if not cache.exists():
        raise HTTPException(
            status_code=404,
            detail="Knowledge graph not built yet. Run POST /api/link first.",
        )
    return KnowledgeGraph.load(cache)
