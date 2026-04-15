"""Hypothesis generator: calls Claude to produce structured cross-domain hypotheses."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from nexuslink.llm.prompts.templates import get_system_prompt, render_template
from nexuslink.utils.json_parser import extract_json
from nexuslink.wiki.graph.builder import KnowledgeGraph, _sanitize as _sanitize_name
from nexuslink.wiki.linker.bridge_finder import ConceptBridge

if TYPE_CHECKING:
    from nexuslink.config import NexusConfig

_WIKI_DIR = Path(__file__).parent.parent.parent / "wiki"
_HYPOTHESES_DIR = _WIKI_DIR / "03-hypotheses"

_MODEL = "claude-sonnet-4-20250514"
_BATCH_SIZE = 5            # max bridges per API call (per domain-pair group)
_MAX_TOKENS_GEN = 4096
_SEMAPHORE_LIMIT = 3       # max concurrent Claude calls
_RETRY_DELAYS = [1.0, 2.0, 4.0]   # exponential backoff (seconds)

# Re-export for backward-compat with existing callers / tests
__all__ = [
    "GeneratedHypothesis",
    "HypothesisGenerator",
    "extract_json",     # re-exported from utils
    "_parse_hypothesis_list",
]


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class GeneratedHypothesis(BaseModel):
    """A hypothesis produced by the LLM generation stage."""

    id: str = Field(default_factory=lambda: f"H{str(uuid.uuid4())[:8].upper()}")
    wiki_id: str = ""  # set to "H001", "H002", ... after note is written
    statement: str
    evidence_bridges: list[str] = Field(default_factory=list)
    domains_spanned: list[str] = Field(default_factory=list)
    suggested_experiments: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    mechanistic_depth: float = Field(ge=0.0, le=10.0, default=0.0)
    falsifiability_score: float = Field(ge=0.0, le=10.0, default=0.0)
    raw_reasoning: str = ""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HypothesisGenerator:
    """Generates structured hypotheses from cross-domain concept bridges via Claude."""

    def __init__(
        self,
        config: "NexusConfig | None" = None,
        *,
        api_key: str | None = None,
    ) -> None:
        import os

        self._ollama_model = (
            (config.ollama_model if config else None)
            or os.environ.get("OLLAMA_MODEL")
        )

        if self._ollama_model:
            logger.info(f"Using local Ollama model: {self._ollama_model}")
            self._client = None
            self._model = self._ollama_model
        else:
            import anthropic
            effective_key = (config.anthropic_api_key if config else None) or api_key
            if not effective_key:
                logger.warning(
                    "No Anthropic API key set. "
                    "Set ANTHROPIC_API_KEY, or set OLLAMA_MODEL to use local inference."
                )
            self._client = anthropic.AsyncAnthropic(api_key=effective_key)
            self._model = _MODEL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        bridges: list[ConceptBridge],
        graph: KnowledgeGraph,
    ) -> list[GeneratedHypothesis]:
        """Generate hypotheses for all *bridges*, writing a Hypothesis.md per result.

        Bridges are grouped by domain pair (max *_BATCH_SIZE* per call) and
        processed concurrently under a semaphore of size *_SEMAPHORE_LIMIT*.
        Near-duplicate hypotheses (cosine sim > 0.9) are removed before return.
        """
        if not bridges:
            logger.warning("No bridges provided; returning empty hypothesis list.")
            return []

        domains = sorted({b.domain_a for b in bridges} | {b.domain_b for b in bridges})
        total_concepts = graph.node_count("concept")

        groups = _group_bridges(bridges, _BATCH_SIZE)
        logger.info(
            "Generating hypotheses from {} bridges across {} groups (model={})",
            len(bridges), len(groups), self._model,
        )

        sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)

        async def _bounded(group: list[ConceptBridge]) -> list[GeneratedHypothesis]:
            async with sem:
                return await self._generate_group(group, domains, total_concepts)

        batch_results = await asyncio.gather(
            *[_bounded(g) for g in groups],
            return_exceptions=True,
        )

        all_hypotheses: list[GeneratedHypothesis] = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error("Group {} generation failed: {}", i, result)
            else:
                all_hypotheses.extend(result)

        unique = await self._deduplicate(all_hypotheses)

        # Assign stable wiki IDs before writing notes
        _HYPOTHESES_DIR.mkdir(parents=True, exist_ok=True)
        for idx, h in enumerate(unique):
            h.wiki_id = f"H{idx + 1:03d}"
        await asyncio.gather(
            *[self._write_wiki_note(h) for h in unique]
        )

        logger.info("Generated {} hypotheses ({} unique)", len(all_hypotheses), len(unique))
        return unique

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int = _MAX_TOKENS_GEN,
    ) -> dict | list:
        """Call Claude or Ollama with backoff retry, parse and return JSON."""
        if getattr(self, "_ollama_model", None):
            return await self._call_ollama(system, user, max_tokens)

        import anthropic
        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, 1):
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = resp.content[0].text
                try:
                    logger.debug(
                        "Claude usage: input={} output={} tokens",
                        resp.usage.input_tokens,
                        resp.usage.output_tokens,
                    )
                except AttributeError:
                    pass
                return extract_json(text)
            except anthropic.RateLimitError as exc:
                last_exc = exc
                logger.warning("Rate limit (attempt {}), retrying in {}s", attempt, delay)
                await asyncio.sleep(delay)
            except anthropic.APIStatusError as exc:
                last_exc = exc
                logger.error("API error on attempt {}: {} {}", attempt, exc.status_code, exc.message)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
            except Exception as exc:
                last_exc = exc
                logger.error("Unexpected error on attempt {}: {}", attempt, exc)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Claude API failed after {len(_RETRY_DELAYS)} attempts"
        ) from last_exc

    async def _call_ollama(
        self,
        system: str,
        user: str,
        max_tokens: int,
    ) -> dict | list:
        import os
        import httpx
        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
        payload = {
            "model": self._ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": max_tokens}
        }
        
        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, 1):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    text = resp.json()["message"]["content"]
                    return extract_json(text)
            except Exception as exc:
                last_exc = exc
                logger.error("Ollama API error on attempt {}: {}", attempt, exc)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                    
        raise RuntimeError(
            f"Ollama API failed after {len(_RETRY_DELAYS)} attempts"
        ) from last_exc

    async def _generate_group(
        self,
        group: list[ConceptBridge],
        domains: list[str],
        total_concepts: int,
    ) -> list[GeneratedHypothesis]:
        system = get_system_prompt("hypothesis_generation")
        user = render_template(
            "hypothesis_generation",
            bridges=group,
            domains=domains,
            total_concepts=total_concepts,
        )
        data = await self._call_llm(system, user)
        return _parse_hypothesis_list(data, group)

    async def _deduplicate(
        self,
        hyps: list[GeneratedHypothesis],
    ) -> list[GeneratedHypothesis]:
        """Remove hypotheses with cosine similarity > 0.9 to an already-kept one."""
        if len(hyps) <= 1:
            return hyps

        import numpy as np

        from nexuslink.raw.schemas.models import ExtractedEntity  # noqa: PLC0415
        from nexuslink.wiki.linker.embedder import ConceptEmbedder  # noqa: PLC0415

        entities = [
            ExtractedEntity(
                name=h.statement[:200],
                entity_type="phenomenon",
                source_doc_id=h.id,
                context_sentence="",
            )
            for h in hyps
        ]

        try:
            embedder = ConceptEmbedder()
            emb_map: dict[str, np.ndarray] = await asyncio.to_thread(
                embedder.embed_batch, entities
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Deduplication embedding failed ({}), skipping", exc)
            return hyps

        unique: list[GeneratedHypothesis] = []
        unique_embs: list[np.ndarray] = []

        for h, ent in zip(hyps, entities):
            emb = emb_map.get(ent.name)
            if emb is None:
                unique.append(h)
                continue
            if all(float(np.dot(emb, u)) <= 0.9 for u in unique_embs):
                unique.append(h)
                unique_embs.append(emb)

        dropped = len(hyps) - len(unique)
        if dropped:
            logger.info("Deduplication removed {} near-duplicate hypothesis(es)", dropped)
        return unique

    async def _write_wiki_note(self, hyp: GeneratedHypothesis) -> None:
        """Write a Hypothesis.md note in wiki/03-hypotheses/.

        Evidence bridge format: [[entity_a]] ↔ [[entity_b]] — two separate
        wikilinks so Obsidian can resolve each concept independently.
        novelty_score is left as a placeholder; pipeline.py fills it in after
        ranking via update_wiki_note_scores().
        """
        domains_yaml = "[" + ", ".join(f'"{d}"' for d in hyp.domains_spanned) + "]"

        # Each bridge key is "entity_a::entity_b" — render as two wikilinks
        # _sanitize_name ensures the link target matches the actual concept note filename
        bridge_links = "\n".join(
            f"- [[{_sanitize_name(k.split('::', 1)[0])}]] ↔ [[{_sanitize_name(k.split('::', 1)[1])}]]"
            if "::" in k else f"- [[{_sanitize_name(k)}]]"
            for k in hyp.evidence_bridges
        )

        experiments_md = "\n".join(
            f"{i + 1}. {exp}" for i, exp in enumerate(hyp.suggested_experiments)
        )

        # Cross-domain bridge: first bridge as a single prominent link pair
        first = next((k for k in hyp.evidence_bridges if "::" in k), None)
        if first:
            a, b = first.split("::", 1)
            cross_domain = f"[[{_sanitize_name(a)}]] ↔ [[{_sanitize_name(b)}]]"
        else:
            cross_domain = "<!-- see evidence above -->"

        content = f"""\
---
id: "{hyp.wiki_id}"
confidence: {hyp.confidence:.2f}
novelty_score: 0.0
feasibility_score: 0.0
impact_score: 0.0
mechanistic_depth: 0.0
falsifiability_score: 0.0
composite_score: 0.0
domains_spanned: {domains_yaml}
status: generated
tags: []
---

## Hypothesis Statement

{hyp.statement}

## Evidence From

{bridge_links or "<!-- none yet -->"}

## Cross-Domain Bridge

{cross_domain}

## Suggested Experiments

{experiments_md or "<!-- none yet -->"}

## References

## Related Hypotheses (live)

```dataview
LIST composite_score
FROM "03-hypotheses"
WHERE file.name != this.file.name
SORT composite_score DESC
LIMIT 10
```
"""
        path = _HYPOTHESES_DIR / f"{hyp.wiki_id}.md"
        await asyncio.to_thread(path.write_text, content, "utf-8")
        logger.debug("Wrote {}", path)


# ---------------------------------------------------------------------------
# Post-ranking note updater
# ---------------------------------------------------------------------------

async def update_wiki_note_scores(scored: Any) -> None:
    """Patch a hypothesis note's frontmatter with scores from the ranking step.

    Called after HypothesisRanker.refine_top_n() so every Hypothesis.md
    reflects the final novelty/feasibility/impact scores, not just placeholders.
    Works by text replacement on the frontmatter lines — no YAML parser needed.
    """
    import re as _re

    if not getattr(scored, "wiki_id", None):
        return
    path = _HYPOTHESES_DIR / f"{scored.wiki_id}.md"
    if not path.exists():
        return

    content = await asyncio.to_thread(path.read_text, "utf-8")
    content = _re.sub(r"^novelty_score:.*$", f"novelty_score: {scored.novelty_score:.1f}", content, flags=_re.MULTILINE)
    content = _re.sub(r"^feasibility_score:.*$", f"feasibility_score: {scored.feasibility_score:.1f}", content, flags=_re.MULTILINE)
    content = _re.sub(r"^impact_score:.*$", f"impact_score: {scored.impact_score:.1f}", content, flags=_re.MULTILINE)
    mech = getattr(scored, "mechanistic_depth", 0.0)
    fals = getattr(scored, "falsifiability_score", 0.0)
    comp = getattr(scored, "composite_score", 0.0)
    if _re.search(r"^mechanistic_depth:.*$", content, flags=_re.MULTILINE):
        content = _re.sub(r"^mechanistic_depth:.*$", f"mechanistic_depth: {mech:.1f}", content, flags=_re.MULTILINE)
    else:
        content = _re.sub(r"^impact_score:.*$", f"impact_score: {scored.impact_score:.1f}\nmechanistic_depth: {mech:.1f}", content, flags=_re.MULTILINE)
    if _re.search(r"^falsifiability_score:.*$", content, flags=_re.MULTILINE):
        content = _re.sub(r"^falsifiability_score:.*$", f"falsifiability_score: {fals:.1f}", content, flags=_re.MULTILINE)
    else:
        content = _re.sub(r"^mechanistic_depth:.*$", f"mechanistic_depth: {mech:.1f}\nfalsifiability_score: {fals:.1f}", content, flags=_re.MULTILINE)
    if _re.search(r"^composite_score:.*$", content, flags=_re.MULTILINE):
        content = _re.sub(r"^composite_score:.*$", f"composite_score: {comp:.2f}", content, flags=_re.MULTILINE)
    else:
        content = _re.sub(r"^falsifiability_score:.*$", f"falsifiability_score: {fals:.1f}\ncomposite_score: {comp:.2f}", content, flags=_re.MULTILINE)
    content = _re.sub(r"^status:.*$", "status: scored", content, flags=_re.MULTILINE)
    await asyncio.to_thread(path.write_text, content, "utf-8")
    logger.debug("Updated scores for {}", scored.wiki_id)


# ---------------------------------------------------------------------------
# Parsing helpers (module-level for testability)
# ---------------------------------------------------------------------------

def _parse_hypothesis_list(
    raw: str | list | dict,
    bridges: list[ConceptBridge],
) -> list[GeneratedHypothesis]:
    """Parse Claude's response (text or pre-parsed JSON) into GeneratedHypothesis objects."""
    if isinstance(raw, str):
        data = extract_json(raw)
    elif isinstance(raw, dict):
        data = [raw]
    else:
        data = raw

    if not isinstance(data, list):
        data = [data]

    all_bridge_keys = [f"{b.entity_a}::{b.entity_b}" for b in bridges]
    hypotheses: list[GeneratedHypothesis] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        statement = item.get("statement", "").strip()
        if not statement:
            continue
        # Use bridge_index from LLM output to assign the specific inspiring bridge.
        # Fall back to all bridges only if the field is absent or out of range.
        idx = item.get("bridge_index")
        if isinstance(idx, int) and 0 <= idx < len(all_bridge_keys):
            evidence_keys = [all_bridge_keys[idx]]
        else:
            evidence_keys = all_bridge_keys
        hypotheses.append(
            GeneratedHypothesis(
                statement=statement,
                evidence_bridges=evidence_keys,
                domains_spanned=item.get("domains_spanned", []),
                suggested_experiments=_coerce_str_list(item.get("suggested_experiments", [])),
                confidence=float(item.get("confidence", 0.5)),
                raw_reasoning=item.get("reasoning", ""),
            )
        )

    return hypotheses


def _coerce_str_list(items: list) -> list[str]:
    """Normalize a list that may contain strings or dicts into a list of strings.

    Ollama sometimes returns structured objects instead of plain strings.
    We flatten dicts by joining their values.
    """
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # prefer 'title' + 'description', fall back to joining all values
            parts = [str(v) for k, v in item.items() if v and k != "id"]
            result.append(" — ".join(parts) if parts else str(item))
        else:
            result.append(str(item))
    return result


def _group_bridges(
    bridges: list[ConceptBridge],
    max_per_group: int,
) -> list[list[ConceptBridge]]:
    """Group bridges by canonical domain-pair key, splitting into max_per_group chunks."""
    pairs: dict[tuple[str, str], list[ConceptBridge]] = defaultdict(list)
    for b in bridges:
        key = (min(b.domain_a, b.domain_b), max(b.domain_a, b.domain_b))
        pairs[key].append(b)

    groups: list[list[ConceptBridge]] = []
    for pair_bridges in pairs.values():
        for i in range(0, len(pair_bridges), max_per_group):
            groups.append(pair_bridges[i : i + max_per_group])

    return groups
