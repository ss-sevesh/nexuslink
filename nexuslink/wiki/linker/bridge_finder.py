"""Bridge finder: detects non-obvious conceptual links between entities from different domains."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from nexuslink.raw.schemas.models import ExtractedEntity
from nexuslink.wiki.linker.embedder import ConceptEmbedder

BridgeType = Literal["analogous", "enables", "extends", "contradicts"]


class ConceptBridge(BaseModel):
    """A detected cross-domain conceptual link between two entities."""

    entity_a: str
    entity_b: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    domain_a: str
    domain_b: str
    bridge_type: BridgeType
    entity_type_a: str = ""
    entity_type_b: str = ""


class BridgeFinder:
    """Finds conceptual bridges between entities drawn from different scientific domains.

    Usage::

        embedder = ConceptEmbedder()
        finder = BridgeFinder(embedder)
        bridges = finder.find_bridges({"physics": phys_entities, "biology": bio_entities})
    """

    def __init__(self, embedder: ConceptEmbedder) -> None:
        self._embedder = embedder

    def find_bridges(
        self,
        entities_by_domain: dict[str, list[ExtractedEntity]],
        threshold: float = 0.65,
    ) -> list[ConceptBridge]:
        """Return cross-domain bridges with cosine similarity ≥ *threshold*.

        Only entity pairs from *different* domains are considered — same-domain
        matches are always filtered out.  Results are sorted by similarity
        descending.
        """
        domains = [d for d, ents in entities_by_domain.items() if ents]
        if len(domains) < 2:
            logger.info("Need 2+ non-empty domains; got {}. Returning no bridges.", len(domains))
            return []

        # Deduplicate entities per domain by name (keep last context seen)
        deduped: dict[str, dict[str, ExtractedEntity]] = {
            domain: {e.name: e for e in entities}
            for domain, entities in entities_by_domain.items()
            if entities
        }

        # Embed each domain's unique entities in one batched call
        embeddings_by_domain: dict[str, dict[str, np.ndarray]] = {}
        for domain, ent_map in deduped.items():
            embeddings_by_domain[domain] = self._embedder.embed_batch(list(ent_map.values()))

        bridges: list[ConceptBridge] = []
        domain_list = list(deduped.keys())

        for i, domain_a in enumerate(domain_list):
            for domain_b in domain_list[i + 1 :]:
                embs_a = embeddings_by_domain[domain_a]
                embs_b = embeddings_by_domain[domain_b]
                ents_a = deduped[domain_a]
                ents_b = deduped[domain_b]

                for name_a, emb_a in embs_a.items():
                    for name_b, emb_b in embs_b.items():
                        # Skip self-loops — same concept name across domains is not a bridge
                        if name_a.lower() == name_b.lower():
                            continue
                        # Skip near-duplicates — one name contained in the other
                        # (e.g. "Axial-ViT-B/32" inside "Axial-ViT-B/16" share the prefix)
                        if _is_near_duplicate(name_a, name_b):
                            continue
                        # Cosine similarity of unit-normalised vectors = dot product
                        sim = float(np.clip(np.dot(emb_a, emb_b), -1.0, 1.0))
                        if sim < threshold:
                            continue

                        type_a = ents_a[name_a].entity_type
                        type_b = ents_b[name_b].entity_type
                        bridges.append(
                            ConceptBridge(
                                entity_a=name_a,
                                entity_b=name_b,
                                similarity_score=round(sim, 4),
                                domain_a=domain_a,
                                domain_b=domain_b,
                                bridge_type=_infer_bridge_type(sim, type_a, type_b),
                                entity_type_a=type_a,
                                entity_type_b=type_b,
                            )
                        )

        bridges.sort(key=lambda b: b.similarity_score, reverse=True)
        logger.info(
            "Found {} bridges across {} domain pairs (threshold={})",
            len(bridges),
            len(domain_list) * (len(domain_list) - 1) // 2,
            threshold,
        )
        return bridges


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

# Strips version/variant suffixes: "/32", "-B16", " v2", ".5", trailing digits
_VERSION_SUFFIX_RE = re.compile(r"[/\-\s\.][a-z]?\d+[a-z]*$", re.IGNORECASE)


def _stem(name: str) -> str:
    """Strip version suffixes iteratively until stable."""
    s = name.lower().strip("./- ")
    prev = None
    while s != prev:
        prev = s
        s = _VERSION_SUFFIX_RE.sub("", s).strip("./- ")
    return s


def _is_near_duplicate(a: str, b: str) -> bool:
    """Return True if a and b are likely the same concept with minor variation.

    Catches cases like:
    - "ViT-B/32" vs "ViT-B/16"         (same model, different resolution)
    - "Axial-ViT-B/32" vs "Axial-ViT-B/16"
    - "COUNTERFACT" vs "COUNTERFACT Composition" (one is prefix of other)
    - "GPT-3" vs "GPT-J."              (same model family via stem match)
    """
    la, lb = a.lower().strip("./- "), b.lower().strip("./- ")

    # Rule 1: one is a prefix of the other (length >= 3)
    shorter, longer = (la, lb) if len(la) <= len(lb) else (lb, la)
    if len(shorter) >= 3 and longer.startswith(shorter):
        return True

    # Rule 2: common prefix >= 6 chars covering >= 55% of the longer name
    # (lowered from 8 to catch "vit-b/32" vs "vit-b/16" — "vit-b/" = 6 chars)
    common = 0
    for ca, cb in zip(la, lb):
        if ca == cb:
            common += 1
        else:
            break
    if common >= 6 and common / max(len(la), len(lb)) >= 0.55:
        return True

    # Rule 3: version-stripped stems are identical (catches "ViT-B/32" vs "ViT-B/16")
    if len(la) >= 4 and _stem(a) == _stem(b) and _stem(a):
        return True

    return False


# ---------------------------------------------------------------------------
# Bridge type inference
# ---------------------------------------------------------------------------

def _infer_bridge_type(sim: float, type_a: str, type_b: str) -> BridgeType:
    """Assign a bridge type from similarity score and entity types.

    Rules (applied in order):
    - Very high similarity (≥0.85) → always *analogous*
    - A method paired with a non-method → *enables*
    - Same entity type at mid-range similarity → *extends*
    - Default → *enables*
    """
    if sim >= 0.85:
        return "analogous"

    method_types = {"method"}
    is_method_a = type_a in method_types
    is_method_b = type_b in method_types

    if is_method_a ^ is_method_b:  # exactly one is a method
        return "enables"

    if type_a == type_b:
        return "extends"

    return "enables"
