"""End-to-end LLM hypothesis pipeline: load graph → generate → rank → validate → report."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from nexuslink.config import NexusConfig
from nexuslink.llm.hypothesis.generator import HypothesisGenerator, update_wiki_note_scores
from nexuslink.llm.reports.writer import ReportWriter
from nexuslink.llm.scoring.ranker import HypothesisRanker
from nexuslink.wiki.citations.manager import CitationManager
from nexuslink.wiki.graph.builder import KnowledgeGraph

_TOP_BRIDGES = 20
_REFINE_N = 3


async def run_hypothesis_pipeline(
    vault_path: Path,
    config: NexusConfig,
    top_bridges: int = _TOP_BRIDGES,
    skip_validation: bool = False,
) -> dict:
    """Run the full hypothesis pipeline end-to-end.

    Steps
    -----
    1. Load knowledge graph from *vault_path/.cache/graph.gpickle*
    2. Select top *top_bridges* cross-domain bridges by similarity
    3. Generate hypotheses via :class:`~nexuslink.llm.hypothesis.generator.HypothesisGenerator`
    4. Rank via :class:`~nexuslink.llm.scoring.ranker.HypothesisRanker`
    5. Refine top *_REFINE_N* hypotheses
    6. Validate claims and citations (unless *skip_validation*)
    7. Generate Markdown + LaTeX report via :class:`~nexuslink.llm.reports.writer.ReportWriter`
    8. Update PROGRESS.md

    Returns a summary dict.
    """
    cache_path = vault_path / ".cache" / "graph.gpickle"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found at {cache_path}. "
            "Run the WIKI linking pipeline (`nexuslink link`) first."
        )

    # ----------------------------------------------------------------
    # 1. Load graph
    # ----------------------------------------------------------------
    kg = await asyncio.to_thread(KnowledgeGraph.load, cache_path)
    logger.info(
        "Graph loaded: {} concepts, {} papers",
        kg.node_count("concept"), kg.node_count("paper"),
    )

    # ----------------------------------------------------------------
    # 2. Select bridges
    # ----------------------------------------------------------------
    all_bridges = kg.get_bridges()
    if not all_bridges:
        logger.warning("No cross-domain bridges in graph. Run `nexuslink link` first.")
        return _empty_summary()

    bridges = all_bridges[:top_bridges]
    logger.info("Using {} / {} bridges for generation", len(bridges), len(all_bridges))

    # ----------------------------------------------------------------
    # 3. Generate hypotheses
    # ----------------------------------------------------------------
    generator = HypothesisGenerator(config=config)
    hypotheses = await generator.generate(bridges, kg)

    if not hypotheses:
        logger.error("No hypotheses generated — check API key and model access.")
        return _empty_summary()
    logger.info("Generated {} hypotheses", len(hypotheses))

    # ----------------------------------------------------------------
    # 4. Rank (critique all concurrently)
    # ----------------------------------------------------------------
    ranker = HypothesisRanker(config=config)
    ranked = await ranker.rank_all(hypotheses)

    # ----------------------------------------------------------------
    # 5. Refine top N
    # ----------------------------------------------------------------
    refined = await ranker.refine_top_n(ranked, n=_REFINE_N)

    # Update hypothesis notes with final scores from ranking
    await asyncio.gather(*[update_wiki_note_scores(h) for h in refined])

    # ----------------------------------------------------------------
    # 6. Validate (optional)
    # ----------------------------------------------------------------
    contradictions_total = 0
    citation_issues_total = 0

    if not skip_validation:
        from nexuslink.llm.validation.checker import ClaimChecker  # noqa: PLC0415

        checker = ClaimChecker(config=config)

        contradiction_results = await asyncio.gather(
            *[checker.check_contradictions(h, kg) for h in refined[:_REFINE_N]],
            return_exceptions=True,
        )
        for res in contradiction_results:
            if isinstance(res, list):
                contradictions_total += len(res)

        citation_results = await asyncio.gather(
            *[checker.verify_citations(h, vault_path) for h in refined[:_REFINE_N]],
            return_exceptions=True,
        )
        for res in citation_results:
            if isinstance(res, list):
                citation_issues_total += len(res)

    # ----------------------------------------------------------------
    # 7. Generate report
    # ----------------------------------------------------------------
    citation_mgr = CitationManager()
    writer = ReportWriter(config=config)

    pipeline_stats = {
        "papers_processed": kg.node_count("paper"),
        "total_concepts": kg.node_count("concept"),
        "total_bridges": len(all_bridges),
    }
    report_path = await writer.generate_report(
        refined, kg, citation_mgr,
        vault_path=vault_path,
        pipeline_stats=pipeline_stats,
    )

    # ----------------------------------------------------------------
    # 8. Update PROGRESS.md
    # ----------------------------------------------------------------
    await _mark_phase3_done(vault_path)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    top = refined[0] if refined else None
    summary = {
        "hypotheses_generated": len(hypotheses),
        "hypotheses_refined": len(refined),
        "top_hypothesis": top.statement if top else None,
        "top_composite_score": round(top.composite_score, 2) if top else None,
        "domains_covered": len({b.domain_a for b in bridges} | {b.domain_b for b in bridges}),
        "contradictions_found": contradictions_total,
        "citation_issues_found": citation_issues_total,
        "report_path": report_path,
    }
    logger.info("Pipeline complete: {}", summary)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_summary() -> dict:
    return {
        "hypotheses_generated": 0,
        "hypotheses_refined": 0,
        "top_hypothesis": None,
        "top_composite_score": None,
        "domains_covered": 0,
        "contradictions_found": 0,
        "citation_issues_found": 0,
        "report_path": None,
    }


async def _mark_phase3_done(vault_path: Path) -> None:
    """Update PROGRESS.md to reflect Phase 3 completion."""
    progress = vault_path.parent / "PROGRESS.md"
    if not progress.exists():
        return

    content = await asyncio.to_thread(progress.read_text, "utf-8")

    replacements = [
        ("- [ ] Hypothesis generation prompts", "- [x] Hypothesis generation prompts"),
        ("- [ ] Novelty scoring algorithm", "- [x] Novelty scoring algorithm"),
        ("- [ ] Report generator (LaTeX + MD)", "- [x] Report generator (LaTeX + MD)"),
        ("- [ ] Validation pipeline", "- [x] Validation pipeline"),
    ]
    for old, new in replacements:
        content = content.replace(old, new)

    await asyncio.to_thread(progress.write_text, content, "utf-8")
    logger.debug("Updated PROGRESS.md")
