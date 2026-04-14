"""Claim checker: detects contradictions and citation issues in generated hypotheses."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger
from pydantic import BaseModel

from nexuslink.llm.hypothesis.generator import GeneratedHypothesis, _RETRY_DELAYS
from nexuslink.utils.json_parser import extract_json
from nexuslink.wiki.graph.builder import KnowledgeGraph

if TYPE_CHECKING:
    from nexuslink.config import NexusConfig

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Contradiction(BaseModel):
    claim: str
    contradicting_source: str
    severity: Literal["low", "medium", "high"]


class CitationIssue(BaseModel):
    claim: str
    referenced_paper: str
    issue_type: Literal["not_found", "weak_support", "misattributed"]
    description: str


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class ClaimChecker:
    """Validates hypothesis claims against the knowledge graph and vault notes."""

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
            logger.info("ClaimChecker using local Ollama model: {}", self._ollama_model)
            self._client = None
            self._model = self._ollama_model
        else:
            import anthropic
            effective_key = (config.anthropic_api_key if config else None) or api_key
            if effective_key is None:
                logger.warning(
                    "No Anthropic API key set. "
                    "Set ANTHROPIC_API_KEY or OLLAMA_MODEL in .env."
                )
            self._client = anthropic.AsyncAnthropic(api_key=effective_key)
            self._model = _MODEL

    async def check_contradictions(
        self,
        hypothesis: GeneratedHypothesis,
        graph: KnowledgeGraph,
    ) -> list[Contradiction]:
        """Detect claims in *hypothesis* that contradict concepts in *graph*.

        Uses Claude to semantically compare the hypothesis statement against
        known entity relationships in the knowledge graph.
        """
        concept_summaries = _summarise_graph_concepts(graph, hypothesis)
        if not concept_summaries:
            logger.debug("No relevant concepts in graph for hypothesis {}", hypothesis.id)
            return []

        prompt = _build_contradiction_prompt(hypothesis.statement, concept_summaries)
        raw = await self._call_llm(prompt)
        return _parse_contradictions(raw)

    async def verify_citations(
        self,
        hypothesis: GeneratedHypothesis,
        vault_path: Path,
    ) -> list[CitationIssue]:
        """Check evidence bridges against vault Paper.md files and via Claude.

        For each ``entity_a::entity_b`` bridge:
        1. Checks that at least one Paper.md in *vault_path/papers/* mentions each entity.
        2. Asks Claude whether the bridge plausibly supports the hypothesis.

        Returns a list of :class:`CitationIssue` (empty means no problems detected).
        """
        issues: list[CitationIssue] = []
        papers_dir = vault_path / "papers"

        # --- Vault file check ---
        for bridge_key in hypothesis.evidence_bridges:
            parts = bridge_key.split("::", 1)
            entity_a = parts[0].strip()
            entity_b = parts[1].strip() if len(parts) > 1 else ""

            if entity_a and not await asyncio.to_thread(_find_papers_mentioning, papers_dir, entity_a):
                issues.append(CitationIssue(
                    claim=f"Entity '{entity_a}' in bridge {bridge_key!r}",
                    referenced_paper=entity_a,
                    issue_type="not_found",
                    description=(
                        f"No Paper.md in '{papers_dir}' mentions '{entity_a}'. "
                        "Ingest a paper containing this entity first."
                    ),
                ))

            if entity_b and not await asyncio.to_thread(_find_papers_mentioning, papers_dir, entity_b):
                issues.append(CitationIssue(
                    claim=f"Entity '{entity_b}' in bridge {bridge_key!r}",
                    referenced_paper=entity_b,
                    issue_type="not_found",
                    description=(
                        f"No Paper.md in '{papers_dir}' mentions '{entity_b}'. "
                        "Ingest a paper containing this entity first."
                    ),
                ))

        # --- Semantic check via Claude ---
        if hypothesis.evidence_bridges:
            prompt = _build_citation_prompt(hypothesis)
            raw = await self._call_llm(prompt)
            try:
                data = extract_json(raw)
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        issue_type = item.get("issue_type", "weak_support")
                        if issue_type not in ("not_found", "weak_support", "misattributed"):
                            issue_type = "weak_support"
                        issues.append(CitationIssue(
                            claim=item.get("claim", hypothesis.statement),
                            referenced_paper=item.get("referenced_paper", ""),
                            issue_type=issue_type,  # type: ignore[arg-type]
                            description=item.get("description", ""),
                        ))
            except Exception as exc:
                logger.warning("Citation parse failed for {}: {}", hypothesis.id, exc)

        return issues

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        """Call Claude or Ollama with exponential-backoff retry; return raw text."""
        if self._ollama_model:
            return await self._call_ollama(prompt)
        return await self._call_claude(prompt)

    async def _call_claude(self, prompt: str) -> str:
        import anthropic

        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, 1):
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except anthropic.RateLimitError as exc:
                last_exc = exc
                logger.warning("Rate limit (attempt {}), retrying in {}s", attempt, delay)
                await asyncio.sleep(delay)
            except Exception as exc:
                last_exc = exc
                logger.error("Claude error on attempt {}: {}", attempt, exc)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)

        raise RuntimeError("Claude API failed") from last_exc

    async def _call_ollama(self, prompt: str) -> str:
        import os
        import httpx

        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
        payload = {
            "model": self._ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"num_predict": _MAX_TOKENS},
        }

        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, 1):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json()["message"]["content"]
            except Exception as exc:
                last_exc = exc
                logger.error("Ollama error on attempt {}: {}", attempt, exc)
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)

        raise RuntimeError("Ollama API failed") from last_exc


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _summarise_graph_concepts(
    graph: KnowledgeGraph, hypothesis: GeneratedHypothesis
) -> list[str]:
    """Return text summaries for graph concepts relevant to *hypothesis*."""
    summaries: list[str] = []
    hyp_lower = hypothesis.statement.lower()

    for node_id, attrs in graph._graph.nodes(data=True):
        if attrs.get("type") != "concept":
            continue
        name: str = attrs.get("name", "")
        if not name or name.lower() not in hyp_lower:
            continue
        entity_type = attrs.get("entity_type", "")
        domains = attrs.get("domains", [])
        summaries.append(f"- {name} ({entity_type}, domains: {', '.join(domains)})")

    return summaries[:20]


def _build_contradiction_prompt(statement: str, concept_summaries: list[str]) -> str:
    concepts_text = "\n".join(concept_summaries)
    return f"""\
You are a scientific fact-checker. Evaluate whether the following hypothesis
contradicts any established relationships among the listed scientific concepts.

## Hypothesis

{statement}

## Known Concepts from Knowledge Graph

{concepts_text}

## Task

Identify any direct logical contradictions between the hypothesis and the
established properties/relationships of the listed concepts.
Return only genuine contradictions — not speculative concerns.

Respond with JSON only (empty array if none):

[
  {{
    "claim": "the specific claim in the hypothesis that is contradicted",
    "contradicting_source": "the concept or relationship it contradicts",
    "severity": "low|medium|high"
  }}
]
"""


def _build_citation_prompt(hypothesis: GeneratedHypothesis) -> str:
    bridges_text = "\n".join(
        f"  - {b.replace('::', ' ↔ ')}" for b in hypothesis.evidence_bridges
    )
    return f"""\
You are a scientific citation auditor. Check whether the cited conceptual
bridges genuinely support the stated hypothesis.

## Hypothesis

{hypothesis.statement}

## Evidence Bridges Cited

{bridges_text}

## Task

For each bridge, determine whether it plausibly supports the hypothesis.
Flag only genuine issues (not found, weak support, or misattributed).

Respond with JSON only (empty array if no issues):

[
  {{
    "claim": "the specific claim being made",
    "referenced_paper": "the bridge or source in question",
    "issue_type": "not_found|weak_support|misattributed",
    "description": "brief explanation of the issue"
  }}
]
"""


def _parse_contradictions(raw: str) -> list[Contradiction]:
    try:
        data = extract_json(raw)
        if not isinstance(data, list):
            return []
        contradictions: list[Contradiction] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            severity = item.get("severity", "low")
            if severity not in ("low", "medium", "high"):
                severity = "low"
            contradictions.append(Contradiction(
                claim=item.get("claim", ""),
                contradicting_source=item.get("contradicting_source", ""),
                severity=severity,  # type: ignore[arg-type]
            ))
        return contradictions
    except Exception as exc:
        logger.warning("Contradiction parse error: {}", exc)
        return []


def _find_papers_mentioning(papers_dir: Path, entity: str) -> list[Path]:
    """Return Paper.md files in *papers_dir* whose content contains *entity* (case-insensitive)."""
    if not papers_dir.exists():
        return []
    entity_lower = entity.lower()
    matches: list[Path] = []
    for md_file in papers_dir.glob("*.md"):
        try:
            content = md_file.read_text("utf-8", errors="ignore").lower()
            if entity_lower in content:
                matches.append(md_file)
        except OSError:
            pass
    return matches
