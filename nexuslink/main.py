"""NexusLink — top-level orchestrator for the three-layer research pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from loguru import logger

from nexuslink.config import NexusConfig

# Default vault is the wiki/ directory at the project root (sibling of this file)
_DEFAULT_VAULT = Path(__file__).parent / "wiki"


class NexusLink:
    """Orchestrator for the RAW → WIKI → LLM pipeline.

    Parameters
    ----------
    vault_path:
        Path to the Obsidian vault (``wiki/`` directory).  All pipeline outputs
        are written there.
    config:
        Runtime configuration.  If *None*, a :class:`NexusConfig` is loaded
        from the environment / ``.env`` file.
    """

    def __init__(
        self,
        vault_path: Path | str = _DEFAULT_VAULT,
        config: NexusConfig | None = None,
    ) -> None:
        self._vault = Path(vault_path).resolve()
        self._config = config or NexusConfig()
        self._configure_logging()

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------

    async def ingest(self, source: str) -> dict:
        """Ingest one paper (PDF path, ArXiv ID, or DOI) into the vault.

        Returns a summary dict with ``id``, ``title``, ``source_type``,
        ``entities_found``, and ``wiki_note``.
        """
        from nexuslink.raw.ingestion.pipeline import run_ingestion  # noqa: PLC0415

        logger.info("Ingesting: {}", source)
        result = await run_ingestion(source)
        logger.info(
            "Ingested {!r} — {} entities, note: {}",
            result["title"], result["entities_found"], result["wiki_note"],
        )
        return result

    async def link(self) -> dict:
        """Run the WIKI cross-domain linking pipeline over all papers in the vault.

        Returns a stats dict with ``papers_processed``, ``total_concepts``,
        ``total_bridges``, ``domains_covered``, and ``concept_notes_written``.
        """
        from nexuslink.wiki.linker.pipeline import run_linking  # noqa: PLC0415

        logger.info("Running cross-domain linking (threshold={})", self._config.similarity_threshold)
        stats = await run_linking(threshold=self._config.similarity_threshold)
        logger.info("Linking complete: {}", stats)
        return stats

    async def hypothesize(self, top_n: int | None = None) -> dict:
        """Run the LLM hypothesis generation, scoring, and report pipeline.

        Parameters
        ----------
        top_n:
            Number of top hypotheses to refine and include in the final report.
            Defaults to ``config.top_n_hypotheses``.

        Returns a dict with ``hypotheses_generated``, ``top_hypothesis``,
        ``top_composite_score``, ``report_path``, and more.
        """
        from nexuslink.llm.hypothesis.pipeline import run_hypothesis_pipeline  # noqa: PLC0415

        n = top_n or self._config.top_n_hypotheses
        logger.info("Generating hypotheses (top_n={})", n)
        result = await run_hypothesis_pipeline(
            vault_path=self._vault,
            config=self._config,
            top_bridges=n * 4,
        )
        return result

    async def run_full(self, sources: list[str]) -> str:
        """Chain all three pipeline layers: ingest → link → hypothesize.

        Parameters
        ----------
        sources:
            List of paper sources (PDF paths, ArXiv IDs, or DOIs).

        Returns
        -------
        str
            Path to the generated Markdown report file.
        """
        logger.info("Starting full pipeline on {} source(s)", len(sources))

        # 1. Ingest every source (run concurrently for I/O efficiency)
        ingest_results = await asyncio.gather(
            *[self.ingest(s) for s in sources], return_exceptions=True
        )
        failed = [r for r in ingest_results if isinstance(r, Exception)]
        if failed:
            logger.warning("{} ingestion(s) failed: {}", len(failed), failed)

        succeeded = len(ingest_results) - len(failed)
        logger.info("{}/{} sources ingested successfully", succeeded, len(sources))

        # 2. Link
        await self.link()

        # 3. Hypothesize + report
        result = await self.hypothesize()
        report_path = result.get("report_path") or ""
        logger.info("Full pipeline complete. Report: {}", report_path)
        return report_path

    async def status(self) -> dict:
        """Return current vault statistics.

        Reads the vault with obsidiantools (falls back to filesystem scan if
        obsidiantools is unavailable).  Also loads the cached knowledge graph
        for bridge and domain counts.
        """
        vault = self._vault
        papers_dir = vault / "papers"
        concepts_dir = vault / "concepts"
        hypotheses_dir = vault / "03-hypotheses"
        graph_cache = vault / ".cache" / "graph.gpickle"

        total_papers = _count_md(papers_dir)
        total_concepts = _count_md(concepts_dir)
        total_hypotheses = _count_md(hypotheses_dir)
        total_bridges = 0
        domains_covered: list[str] = []

        if graph_cache.exists():
            try:
                from nexuslink.wiki.graph.builder import KnowledgeGraph  # noqa: PLC0415

                kg = await asyncio.to_thread(KnowledgeGraph.load, graph_cache)
                total_bridges = len(kg.get_bridges())
                domain_set: set[str] = set()
                for _, attrs in kg._graph.nodes(data=True):
                    if attrs.get("type") == "concept":
                        domain_set.update(attrs.get("domains", []))
                domains_covered = sorted(domain_set)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load knowledge graph for status: {}", exc)

        # Enrich with obsidiantools vault-wide note count
        total_vault_notes = total_papers + total_concepts + total_hypotheses
        try:
            import obsidiantools.api as otools  # noqa: PLC0415

            vault_obj = await asyncio.to_thread(lambda: otools.Vault(vault).connect())
            total_vault_notes = len(vault_obj.md_file_index)
        except Exception:  # noqa: BLE001
            pass  # obsidiantools optional; filesystem count is the fallback

        return {
            "total_papers": total_papers,
            "total_concepts": total_concepts,
            "total_bridges": total_bridges,
            "total_hypotheses": total_hypotheses,
            "domains_covered": domains_covered,
            "total_vault_notes": total_vault_notes,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        logger.remove()
        logger.add(
            sys.stderr,
            level=self._config.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
            colorize=True,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _count_md(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("*.md"))
