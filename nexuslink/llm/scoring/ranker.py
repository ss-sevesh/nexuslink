"""Hypothesis ranker: critiques, scores, and refines generated hypotheses via Claude."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import Field, computed_field

from nexuslink.llm.hypothesis.generator import GeneratedHypothesis, _RETRY_DELAYS, _coerce_str_list
from nexuslink.llm.prompts.templates import get_system_prompt, render_template
from nexuslink.utils.json_parser import extract_json

if TYPE_CHECKING:
    from nexuslink.config import NexusConfig

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS_CRITIQUE = 1024
_MAX_TOKENS_REFINE = 2048
_SEMAPHORE_LIMIT = 3


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class ScoredHypothesis(GeneratedHypothesis):
    """A hypothesis that has been critiqued and scored."""

    novelty_score: float = Field(ge=0.0, le=10.0, default=5.0)
    feasibility_score: float = Field(ge=0.0, le=10.0, default=5.0)
    impact_score: float = Field(ge=0.0, le=10.0, default=5.0)
    weaknesses: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    critique_summary: str = ""
    overall_rank: int = 0

    @computed_field  # type: ignore[misc]
    @property
    def composite_score(self) -> float:
        """Weighted composite: 30% novelty + 25% impact + 20% feasibility + 15% mechanistic_depth + 10% falsifiability."""
        return (
            0.30 * self.novelty_score
            + 0.25 * self.impact_score
            + 0.20 * self.feasibility_score
            + 0.15 * self.mechanistic_depth
            + 0.10 * self.falsifiability_score
        )


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class HypothesisRanker:
    """Critiques, ranks, and refines a list of generated hypotheses."""

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

    _SCORE_FIELDS = frozenset({
        "novelty_score", "feasibility_score", "impact_score",
        "mechanistic_depth", "falsifiability_score",
        "weaknesses", "strengths", "critique_summary", "overall_rank",
    })

    async def critique(self, hypothesis: GeneratedHypothesis) -> ScoredHypothesis:
        """Score *hypothesis* on novelty, feasibility, and impact via Claude."""
        system = get_system_prompt("hypothesis_critique")
        user = render_template("hypothesis_critique", hypothesis=hypothesis)

        try:
            data = await self._call_llm(system, user, _MAX_TOKENS_CRITIQUE)
            if not isinstance(data, dict):
                data = {}
        except Exception as exc:
            logger.warning("Critique failed for {}: {}", hypothesis.id, exc)
            data = {}

        # Exclude scored fields so we never pass duplicate kwargs when
        # hypothesis is already a ScoredHypothesis (e.g. during refinement).
        base = hypothesis.model_dump(exclude=self._SCORE_FIELDS)
        return ScoredHypothesis(
            **base,
            novelty_score=float(data.get("novelty_score", 5.0)),
            feasibility_score=float(data.get("feasibility_score", 5.0)),
            impact_score=float(data.get("impact_score", 5.0)),
            mechanistic_depth=float(data.get("mechanistic_depth", 0.0)),
            falsifiability_score=float(data.get("falsifiability_score", 0.0)),
            strengths=_coerce_str_list(data.get("strengths", [])),
            weaknesses=_coerce_str_list(data.get("weaknesses", [])),
            critique_summary=str(data.get("critique_summary", "")),
        )

    async def rank_all(
        self, hypotheses: list[GeneratedHypothesis]
    ) -> list[ScoredHypothesis]:
        """Critique all hypotheses concurrently (semaphore max 3), return sorted list."""
        sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)

        async def _bounded(h: GeneratedHypothesis) -> ScoredHypothesis:
            async with sem:
                return await self.critique(h)

        scored = await asyncio.gather(*[_bounded(h) for h in hypotheses])
        return self._sort_scored(list(scored))

    async def refine_top_n(
        self,
        scored: list[ScoredHypothesis],
        n: int = 3,
    ) -> list[ScoredHypothesis]:
        """Refine the top-*n* hypotheses, re-critique each, return re-sorted list."""
        top = scored[:n]
        rest = scored[n:]

        refined_list = await asyncio.gather(
            *[self._refine_one(h) for h in top],
            return_exceptions=True,
        )

        result: list[ScoredHypothesis] = []
        for original, outcome in zip(top, refined_list):
            if isinstance(outcome, Exception):
                logger.error("Refinement failed for {}: {}", original.id, outcome)
                result.append(original)
            else:
                result.append(outcome)

        return self._sort_scored(result + rest)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int,
    ) -> dict | list:
        """Call Claude or Ollama with backoff retry, parse JSON from response."""
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

    async def _refine_one(self, scored: ScoredHypothesis) -> ScoredHypothesis:
        system = get_system_prompt("hypothesis_refinement")
        user = render_template("hypothesis_refinement", hypothesis=scored, scored=scored)

        data = await self._call_llm(system, user, _MAX_TOKENS_REFINE)
        if not isinstance(data, dict):
            return scored

        raw_exps = data.get("revised_experiments", scored.suggested_experiments)
        updated = scored.model_copy(update={
            "statement": data.get("revised_statement", scored.statement),
            "suggested_experiments": _coerce_str_list(raw_exps) if isinstance(raw_exps, list) else scored.suggested_experiments,
            "confidence": float(data.get("revised_confidence", scored.confidence)),
        })
        logger.debug("Refined hypothesis {}", scored.id)

        # Re-critique so scores reflect the improved statement
        return await self.critique(updated)

    def _sort_scored(self, scored: list[ScoredHypothesis]) -> list[ScoredHypothesis]:
        ranked = sorted(scored, key=lambda h: h.composite_score, reverse=True)
        for i, h in enumerate(ranked):
            h.overall_rank = i + 1
        return ranked
