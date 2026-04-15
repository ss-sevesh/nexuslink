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
    """Run the full pipeline (ingest -> link -> hypothesize) on one or more sources.

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


# ---------------------------------------------------------------------------
# nexuslink cycle
# ---------------------------------------------------------------------------

@cli.command("cycle")
@_VAULT_OPTION
@click.option("--continuous", is_flag=True, default=False,
              help="Run continuously until vault stabilizes.")
@click.option("--interval", default=60, type=int, show_default=True,
              help="Minutes between cycles in continuous mode.")
@click.option("--max-cycles", default=10, type=int, show_default=True,
              help="Maximum number of cycles to run in continuous mode.")
def cmd_cycle(vault_path: str | None, continuous: bool, interval: int, max_cycles: int) -> None:
    """Run one full research cycle (heal -> feedback -> link -> hypothesize -> expand)."""
    from nexuslink.cycle import NexusLinkCycle

    config = NexusConfig()
    effective_vault = Path(vault_path) if vault_path else config.vault_path
    cycle = NexusLinkCycle(vault_path=effective_vault, config=config)

    if continuous:
        click.echo(f"Running continuously (max_cycles={max_cycles}, interval={interval}m)...")
        cycle.run_continuous(interval_minutes=interval, max_cycles=max_cycles)
    else:
        report = cycle.run_cycle()
        click.echo(f"Cycle {report.cycle_number} complete:")
        click.echo(f"  Papers         : {report.papers_before} -> {report.papers_after}")
        click.echo(f"  Concepts       : {report.concepts_before} -> {report.concepts_after}")
        click.echo(f"  Hypotheses     : {report.hypotheses_generated}")
        click.echo(f"  Heal actions   : {report.heal_actions}")
        click.echo(f"  Auto-expanded  : {report.papers_auto_expanded}")
        click.echo(f"  Feedback used  : {report.feedback_applied}")


# ---------------------------------------------------------------------------
# nexuslink heal
# ---------------------------------------------------------------------------

@cli.command("heal")
@_VAULT_OPTION
@click.option("--apply", is_flag=True, default=False,
              help="Apply changes (default is dry-run).")
def cmd_heal(vault_path: str | None, apply: bool) -> None:
    """Heal the vault: merge duplicates, fix broken links, prune low-quality concepts."""
    from wiki.vault.reader import VaultReader
    from wiki.vault.healer import VaultHealer

    config = NexusConfig()
    effective_vault = Path(vault_path) if vault_path else config.vault_path
    reader = VaultReader(effective_vault)
    healer = VaultHealer(reader)
    report = healer.heal(dry_run=not apply)

    mode = "Applied" if apply else "Dry-run"
    click.echo(f"Heal report ({mode}):")
    click.echo(f"  Duplicates merged  : {report.duplicates_merged}")
    click.echo(f"  Links fixed        : {report.links_fixed}")
    click.echo(f"  Concepts pruned    : {report.concepts_pruned}")
    click.echo(f"  Notes updated      : {report.notes_updated}")
    if not apply:
        click.echo("\nRun with --apply to write changes.")


# ---------------------------------------------------------------------------
# nexuslink expand
# ---------------------------------------------------------------------------

@cli.command("expand")
@_VAULT_OPTION
@click.argument("hypothesis_id", required=False, default=None)
def cmd_expand(vault_path: str | None, hypothesis_id: str | None) -> None:
    """Expand the vault by finding supporting/refuting papers via Semantic Scholar.

    HYPOTHESIS_ID: optional hypothesis ID to expand. If omitted, expands top hypotheses.
    """
    from wiki.vault.reader import VaultReader
    from wiki.vault.expander import AutonomousExpander

    config = NexusConfig()
    effective_vault = Path(vault_path) if vault_path else config.vault_path
    reader = VaultReader(effective_vault)
    s2_key = getattr(config, "semantic_scholar_api_key", None)
    expander = AutonomousExpander(reader, api_key=s2_key)

    if hypothesis_id:
        hypotheses = [h for h in reader.read_all_hypotheses() if h.id == hypothesis_id]
        if not hypotheses:
            click.echo(f"Hypothesis '{hypothesis_id}' not found.", err=True)
            sys.exit(1)
        report = expander.expand_vault_for_hypothesis(hypotheses[0])
        click.echo(f"Expanded {hypothesis_id}:")
        click.echo(f"  Papers found     : {report.papers_found}")
        click.echo(f"  Supporting       : {report.supporting}")
        click.echo(f"  Refuting         : {report.refuting}")
    else:
        result = expander.auto_expand_cycle()
        click.echo("Auto-expand complete:")
        click.echo(f"  Hypotheses expanded : {result['hypotheses_expanded']}")
        click.echo(f"  Papers found        : {result['papers_found']}")
        click.echo(f"  Papers ingested     : {result['papers_ingested']}")
        if result["suggested_domains"]:
            click.echo(f"  Suggested domains   : {', '.join(result['suggested_domains'])}")


# ---------------------------------------------------------------------------
# nexuslink feedback
# ---------------------------------------------------------------------------

@cli.command("feedback")
@_VAULT_OPTION
def cmd_feedback(vault_path: str | None) -> None:
    """Show feedback loop state: reviewed/rejected counts, calibration, few-shot examples."""
    from wiki.vault.reader import VaultReader
    from wiki.vault.feedback import FeedbackLoop

    config = NexusConfig()
    effective_vault = Path(vault_path) if vault_path else config.vault_path
    reader = VaultReader(effective_vault)
    loop = FeedbackLoop(reader)

    reviewed = loop.collect_reviewed_hypotheses()
    rejected = loop.collect_rejected_hypotheses()
    calibration = loop.load_calibration()
    few_shot = loop.build_few_shot_examples()

    click.echo("Feedback Loop State:")
    click.echo(f"  Reviewed/validated hypotheses : {len(reviewed)}")
    click.echo(f"  Rejected hypotheses           : {len(rejected)}")
    click.echo(f"  Few-shot examples (positive)  : {len(few_shot.positive_examples)}")
    click.echo(f"  Few-shot examples (negative)  : {len(few_shot.negative_examples)}")
    click.echo("\nScoring Calibration:")
    click.echo(f"  novelty_bias            : {calibration.novelty_bias:+.3f}")
    click.echo(f"  feasibility_bias        : {calibration.feasibility_bias:+.3f}")
    click.echo(f"  impact_bias             : {calibration.impact_bias:+.3f}")
    click.echo(f"  mechanistic_depth_bias  : {calibration.mechanistic_depth_bias:+.3f}")
    click.echo(f"  falsifiability_bias     : {calibration.falsifiability_bias:+.3f}")


# ---------------------------------------------------------------------------
# nexuslink integrity
# ---------------------------------------------------------------------------

@cli.command("integrity")
@_VAULT_OPTION
def cmd_integrity(vault_path: str | None) -> None:
    """Check evidence integrity of all hypotheses (retraction flags, reliability scores)."""
    from wiki.vault.integrity import EvidenceIntegrityChecker

    config = NexusConfig()
    effective_vault = Path(vault_path) if vault_path else config.vault_path
    checker = EvidenceIntegrityChecker(effective_vault)
    results = checker.check_all_hypotheses()

    click.echo(f"{'Hypothesis':<20} {'Integrity':>10} {'Retractions':>12} {'Issues':>8}")
    click.echo("-" * 54)
    for r in results:
        retracted = len([x for x in r.retraction_flags if x.is_retracted])
        click.echo(
            f"{r.hypothesis_id:<20} {r.overall_integrity_score:>10.2f} {retracted:>12} {len(r.citation_issues):>8}"
        )

    summary = checker.get_integrity_summary()
    click.echo("")
    click.echo("Summary:")
    click.echo(f"  Hypotheses checked           : {summary.get('total_hypotheses_checked', 0)}")
    click.echo(f"  Average integrity score      : {summary.get('average_integrity_score', 0):.3f}")
    click.echo(f"  With retraction flags        : {summary.get('hypotheses_with_retraction_flags', 0)}")
    click.echo(f"  Clean hypotheses             : {summary.get('clean_hypotheses', 0)}")


# ---------------------------------------------------------------------------
# nexuslink benchmark
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--cycles", default=3, help="Number of cycles to run")
@click.option("--export-latex", is_flag=True, help="Export LaTeX table for paper")
def benchmark(cycles, export_latex):
    """Run one-shot vs cyclical benchmark comparison."""
    from nexuslink.benchmark import HypothesisBenchmark
    config = NexusConfig()
    vault_path = config.vault_path
    bench = HypothesisBenchmark(vault_path)
    if export_latex:
        click.echo(bench.export_for_paper())
    else:
        report = bench.run_full_benchmark(n_cycles=cycles)
        click.echo(f"\nImprovement: {report.improvement_pct:+.1f}%")
        click.echo(f"Report saved to wiki/04-reports/")
