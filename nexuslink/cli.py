"""NexusLink command-line interface — entry point: `nexuslink`."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from nexuslink.config import NexusConfig
from nexuslink.main import NexusLink

# Shared option — applied to all sub-commands via context object
_VAULT_OPTION = click.option(
    "--vault-path",
    default=None,
    envvar="NEXUSLINK_VAULT_PATH",
    type=click.Path(),
    help="Path to the Obsidian vault (wiki/ dir). Defaults to ./wiki/.",
    show_default=True,
)


def _nexuslink(vault_path: str | None) -> NexusLink:
    """Construct a NexusLink instance, resolving vault_path from config if unset."""
    config = NexusConfig()
    effective_vault = Path(vault_path) if vault_path else config.vault_path
    return NexusLink(vault_path=effective_vault, config=config)


def _run(coro):
    """Run an async coroutine in the sync click context."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="nexuslink")
def cli() -> None:
    """NexusLink - Cross-Domain Research Hypothesis Engine.

    \b
    Three-layer pipeline:
      1. ingest      - parse papers into the Obsidian vault
      2. link        - build a cross-domain knowledge graph
      3. hypothesize - generate ranked hypotheses with an LLM
    """


# ---------------------------------------------------------------------------
# nexuslink ingest <source>
# ---------------------------------------------------------------------------

@cli.command("ingest")
@click.argument("source")
@_VAULT_OPTION
def cmd_ingest(source: str, vault_path: str | None) -> None:
    """Ingest a single paper into the vault.

    SOURCE may be a local PDF path, an ArXiv ID (e.g. 2101.03961), or a DOI.
    """
    nx = _nexuslink(vault_path)
    result = _run(nx.ingest(source))
    click.echo(f"Ingested: {result['title']!r}")
    click.echo(f"  Entities found : {result['entities_found']}")
    click.echo(f"  Wiki note      : {result['wiki_note']}")


# ---------------------------------------------------------------------------
# nexuslink ingest-batch <folder>
# ---------------------------------------------------------------------------

@cli.command("ingest-batch")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@_VAULT_OPTION
@click.option("--glob", "glob_pattern", default="*.pdf", show_default=True,
              help="Glob pattern for files to ingest.")
def cmd_ingest_batch(folder: Path, vault_path: str | None, glob_pattern: str) -> None:
    """Ingest all papers in FOLDER matching --glob into the vault."""
    pdfs = sorted(folder.glob(glob_pattern))
    if not pdfs:
        click.echo(f"No files matching '{glob_pattern}' found in {folder}.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(pdfs)} file(s) to ingest...")
    nx = _nexuslink(vault_path)

    async def _batch():
        results, errors = [], []
        for pdf in pdfs:
            try:
                r = await nx.ingest(str(pdf))
                results.append(r)
                click.echo(f"  OK {r['title']!r} ({r['entities_found']} entities)")
            except Exception as exc:  # noqa: BLE001
                errors.append((str(pdf), exc))
                click.echo(f"  FAIL {pdf.name}: {exc}", err=True)
        return results, errors

    results, errors = _run(_batch())
    click.echo(f"\nDone: {len(results)} ingested, {len(errors)} failed.")


# ---------------------------------------------------------------------------
# nexuslink link
# ---------------------------------------------------------------------------

@cli.command("link")
@_VAULT_OPTION
@click.option("--threshold", default=None, type=float,
              help="Cosine similarity threshold for bridge detection (0.0–1.0).")
@click.option("--force-rebuild", is_flag=True, default=False,
              help="Ignore cached graph and rebuild from scratch.")
def cmd_link(vault_path: str | None, threshold: float | None, force_rebuild: bool) -> None:
    """Run cross-domain concept linking over all papers in the vault."""
    config = NexusConfig()
    effective_threshold = threshold if threshold is not None else config.similarity_threshold
    nx = _nexuslink(vault_path)

    click.echo(f"Linking vault (threshold={effective_threshold:.2f})...")
    from nexuslink.wiki.linker.pipeline import run_linking  # noqa: PLC0415

    stats = _run(run_linking(threshold=effective_threshold, force_rebuild=force_rebuild))

    click.echo(f"\nLinking complete:")
    click.echo(f"  Papers processed     : {stats.get('papers_processed', 0)}")
    click.echo(f"  Concepts in graph    : {stats.get('total_concepts', 0)}")
    click.echo(f"  Cross-domain bridges : {stats.get('total_bridges', 0)}")
    click.echo(f"  Domains covered      : {stats.get('domains_covered', 0)}")
    click.echo(f"  Concept notes written: {stats.get('concept_notes_written', 0)}")


# ---------------------------------------------------------------------------
# nexuslink hypothesize
# ---------------------------------------------------------------------------

@cli.command("hypothesize")
@_VAULT_OPTION
@click.option("--top-n", default=None, type=int,
              help="Number of top hypotheses to generate and refine.")
@click.option("--skip-validation", is_flag=True, default=False,
              help="Skip contradiction and citation checking (faster).")
def cmd_hypothesize(vault_path: str | None, top_n: int | None, skip_validation: bool) -> None:
    """Generate, rank, and refine hypotheses from the knowledge graph.

    Requires NEXUSLINK_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY) to be set.
    """
    config = NexusConfig()
    effective_n = top_n or config.top_n_hypotheses
    nx = _nexuslink(vault_path)

    click.echo(f"Generating hypotheses (top_n={effective_n})...")
    result = _run(nx.hypothesize(top_n=effective_n))

    click.echo(f"\nHypothesis pipeline complete:")
    click.echo(f"  Generated            : {result.get('hypotheses_generated', 0)}")
    click.echo(f"  Domains covered      : {result.get('domains_covered', 0)}")
    click.echo(f"  Contradictions found : {result.get('contradictions_found', 0)}")
    if result.get("top_hypothesis"):
        click.echo(f"\n  Top hypothesis (score={result.get('top_composite_score'):.2f}):")
        click.echo(f"  {result['top_hypothesis']}")
    if result.get("report_path"):
        click.echo(f"\n  Report: {result['report_path']}")


# ---------------------------------------------------------------------------
# nexuslink run <source1> <source2> ...
# ---------------------------------------------------------------------------

@cli.command("run")
@click.argument("sources", nargs=-1, required=True)
@_VAULT_OPTION
@click.option("--top-n", default=None, type=int,
              help="Number of hypotheses to generate.")
def cmd_run(sources: tuple[str, ...], vault_path: str | None, top_n: int | None) -> None:
    """Run the full pipeline (ingest → link → hypothesize) on one or more sources.

    SOURCES may be PDF paths, ArXiv IDs, or DOIs — mixed types are allowed.
    """
    config = NexusConfig()
    nx = NexusLink(
        vault_path=Path(vault_path) if vault_path else config.vault_path,
        config=config,
    )
    if top_n:
        config.top_n_hypotheses = top_n  # type: ignore[misc]

    click.echo(f"Running full pipeline on {len(sources)} source(s)...")
    report_path = _run(nx.run_full(list(sources)))

    if report_path:
        click.echo(f"\nDone. Report written to: {report_path}")
    else:
        click.echo("\nPipeline finished (no report path returned — check logs).", err=True)


# ---------------------------------------------------------------------------
# nexuslink status
# ---------------------------------------------------------------------------

@cli.command("status")
@_VAULT_OPTION
def cmd_status(vault_path: str | None) -> None:
    """Print current vault statistics."""
    nx = _nexuslink(vault_path)
    stats = _run(nx.status())

    click.echo("NexusLink Vault Status")
    click.echo("=" * 30)
    click.echo(f"  Papers         : {stats['total_papers']}")
    click.echo(f"  Concepts       : {stats['total_concepts']}")
    click.echo(f"  Bridges        : {stats['total_bridges']}")
    click.echo(f"  Hypotheses     : {stats['total_hypotheses']}")
    click.echo(f"  Vault notes    : {stats['total_vault_notes']}")
    if stats["domains_covered"]:
        click.echo(f"  Domains        : {', '.join(stats['domains_covered'])}")
    else:
        click.echo("  Domains        : (run `nexuslink link` to populate)")
