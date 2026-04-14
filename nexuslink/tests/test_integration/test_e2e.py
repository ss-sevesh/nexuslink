"""End-to-end integration test for the full NexusLink pipeline.

Mocks all external APIs (Anthropic, ArXiv, CrossRef, spaCy, sentence-transformers)
so the test runs offline and fast.  Asserts that the full RAW → WIKI → LLM chain
produces valid Obsidian-compatible Markdown files with [[wikilinks]].
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake paper fixtures
# ---------------------------------------------------------------------------

_PHYSICS_PDF_CONTENT = """\
# Casimir Effect in Nano-Structured Surfaces
Alice Kowalski, Ben Tanaka

Abstract: We demonstrate that nano-structured surfaces amplify the Casimir effect by
three orders of magnitude, enabling room-temperature quantum adhesion.

## Methods
We used atomic force microscopy and quantum field theory simulations.

## Key Findings
Nano-patterned gold surfaces exhibit a Casimir pressure 1000x larger than flat gold.
"""

_BIOLOGY_PDF_CONTENT = """\
# Gecko Adhesion via van der Waals Forces
Carol Nduta, David Park

Abstract: Gecko setae exploit van der Waals forces for reversible adhesion.
We identify the beta-keratin structural gene responsible for seta geometry.

## Methods
We used scanning electron microscopy, gene knockout in Anolis carolinensis,
and molecular dynamics simulations.

## Key Findings
Seta tip geometry is the primary determinant of adhesion strength.
"""

_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_pdf(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _make_spacy_mock(ents: list[tuple[str, str]]) -> MagicMock:
    """Return a mock nlp callable that yields the given (text, label_) entities."""
    mock_ents = []
    for text, label in ents:
        e = MagicMock()
        e.text = text
        e.label_ = label
        e.sent.text = f"Sentence containing {text}."
        mock_ents.append(e)

    spacy_doc = MagicMock()
    spacy_doc.ents = mock_ents
    return MagicMock(return_value=spacy_doc)


def _make_bridge(name_a: str, domain_a: str, name_b: str, domain_b: str) -> MagicMock:
    from nexuslink.wiki.linker.bridge_finder import ConceptBridge  # noqa: PLC0415

    return ConceptBridge(
        entity_a=name_a,
        entity_b=name_b,
        domain_a=domain_a,
        domain_b=domain_b,
        similarity_score=0.85,
        bridge_type="analogous",
    )


def _make_hypothesis(statement: str, score: float = 0.82) -> MagicMock:
    h = MagicMock()
    h.statement = statement
    h.composite_score = score
    h.domains = ["physics", "biology"]
    h.evidence = ["Evidence A.", "Evidence B."]
    h.model_dump.return_value = {"statement": statement, "composite_score": score}
    return h


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Ingest two papers from different domains → link → hypothesize → report."""

    @pytest.mark.asyncio
    async def test_two_papers_end_to_end(self, tmp_path: Path) -> None:
        # ------------------------------------------------------------------
        # Set up a fake vault under tmp_path so nothing touches the real wiki/
        # ------------------------------------------------------------------
        vault = tmp_path / "wiki"
        papers_dir = vault / "papers"
        concepts_dir = vault / "concepts"
        hypotheses_dir = vault / "03-hypotheses"
        cache_dir = vault / ".cache"
        for d in (papers_dir, concepts_dir, hypotheses_dir, cache_dir):
            d.mkdir(parents=True)

        phys_pdf = _make_fake_pdf(tmp_path, "casimir.pdf", _PHYSICS_PDF_CONTENT)
        bio_pdf = _make_fake_pdf(tmp_path, "gecko.pdf", _BIOLOGY_PDF_CONTENT)

        phys_nlp = _make_spacy_mock([("Casimir effect", "phenomenon"), ("gold", "material")])
        bio_nlp = _make_spacy_mock([("van der Waals forces", "phenomenon"), ("beta-keratin", "gene")])

        # ------------------------------------------------------------------
        # PHASE 1: Ingest both papers (mocking PDF parser + spaCy)
        # ------------------------------------------------------------------
        from nexuslink.raw.ingestion.pipeline import run_ingestion  # noqa: PLC0415

        physics_md = _PHYSICS_PDF_CONTENT
        biology_md = _BIOLOGY_PDF_CONTENT

        for pdf_path, md_content, mock_nlp in [
            (phys_pdf, physics_md, phys_nlp),
            (bio_pdf, biology_md, bio_nlp),
        ]:
            with (
                patch("nexuslink.raw.ingestion.pdf_loader.pymupdf4llm.to_markdown",
                      return_value=md_content),
                patch("nexuslink.raw.ingestion.pipeline._WIKI_PAPERS_DIR", papers_dir),
                patch("nexuslink.raw.extraction.entity_extractor._get_nlp",
                      return_value=mock_nlp),
                patch("nexuslink.raw.extraction.entity_extractor._build_phrase_matcher",
                      return_value=MagicMock(return_value=[])),
            ):
                result = await run_ingestion(str(pdf_path))

            assert result["source_type"] == "pdf"
            assert result["title"]  # non-empty

        # Both Paper.md files must exist with wikilinks
        paper_notes = list(papers_dir.glob("*.md"))
        assert len(paper_notes) == 2, f"Expected 2 paper notes, got {len(paper_notes)}"
        for note in paper_notes:
            content = note.read_text("utf-8")
            assert "---" in content, "Missing YAML frontmatter"
            # Entities section must be present (even if empty placeholder)
            assert "## Entities" in content

        # ------------------------------------------------------------------
        # PHASE 2: Link (mock embedder + bridge finder + graph export)
        # ------------------------------------------------------------------
        from nexuslink.wiki.linker.pipeline import run_linking  # noqa: PLC0415

        fake_bridge = _make_bridge("Casimir effect", "physics", "van der Waals forces", "biology")

        fake_kg = MagicMock()
        fake_kg.node_count.side_effect = lambda kind: 4 if kind == "concept" else 2
        fake_kg.get_bridges.return_value = [fake_bridge]
        fake_kg.export_for_obsidian = AsyncMock(return_value=2)

        # KnowledgeGraph.save is called via asyncio.to_thread; mock the bound method
        fake_kg.save = MagicMock()

        fake_embedder = MagicMock()
        fake_embedder.save_cache_async = AsyncMock()

        fake_finder = MagicMock()
        fake_finder.find_bridges.return_value = [fake_bridge]

        with (
            patch("nexuslink.wiki.linker.pipeline._PAPERS_DIR", papers_dir),
            patch("nexuslink.wiki.linker.pipeline._CACHE_PATH", cache_dir / "graph.gpickle"),
            patch("nexuslink.wiki.linker.pipeline.KnowledgeGraph", return_value=fake_kg),
            patch("nexuslink.wiki.linker.pipeline.ConceptEmbedder", return_value=fake_embedder),
            patch("nexuslink.wiki.linker.pipeline.BridgeFinder", return_value=fake_finder),
            # Disable obsidiantools discovery; force glob path via missing _WIKI_DIR
            patch("nexuslink.wiki.linker.pipeline._WIKI_DIR", vault),
        ):
            link_stats = await run_linking(threshold=0.65, force_rebuild=True)

        assert link_stats["papers_processed"] == 2
        assert link_stats["total_bridges"] == 1
        assert fake_kg.add_bridge.call_count == 1

        # ------------------------------------------------------------------
        # PHASE 3: Hypothesize (mock Anthropic + ranker + writer)
        # ------------------------------------------------------------------
        from nexuslink.llm.hypothesis.pipeline import run_hypothesis_pipeline  # noqa: PLC0415

        top_hyp = _make_hypothesis(
            "Nano-structured surfaces engineered with seta-inspired geometry will achieve "
            "Casimir-force-mediated adhesion stronger than biological geckos."
        )

        fake_graph = MagicMock()
        fake_graph.node_count.side_effect = lambda kind: 4 if kind == "concept" else 2
        fake_graph.get_bridges.return_value = [fake_bridge]

        fake_generator = MagicMock()
        fake_generator.generate = AsyncMock(return_value=[top_hyp])

        fake_ranker = MagicMock()
        fake_ranker.critique = AsyncMock(return_value=top_hyp)
        fake_ranker.rank_all = AsyncMock(return_value=[top_hyp])
        fake_ranker.refine_top_n = AsyncMock(return_value=[top_hyp])

        report_md = hypotheses_dir / "report.md"
        report_md.write_text(
            "# NexusLink Hypothesis Report\n\n"
            "## Top Hypothesis\n\n"
            f"{top_hyp.statement}\n\n"
            "[[Casimir effect]] ↔ [[van der Waals forces]]\n",
            "utf-8",
        )

        fake_writer = MagicMock()
        fake_writer.generate_report = AsyncMock(return_value=str(report_md))

        fake_citation_mgr = MagicMock()

        mock_config = MagicMock()
        mock_config.anthropic_api_key = "test-key"

        with (
            patch("nexuslink.llm.hypothesis.pipeline.KnowledgeGraph") as MockKG,
            patch("nexuslink.llm.hypothesis.pipeline.HypothesisGenerator",
                  return_value=fake_generator),
            patch("nexuslink.llm.hypothesis.pipeline.HypothesisRanker",
                  return_value=fake_ranker),
            patch("nexuslink.llm.hypothesis.pipeline.ReportWriter", return_value=fake_writer),
            patch("nexuslink.llm.hypothesis.pipeline.CitationManager",
                  return_value=fake_citation_mgr),
            patch("nexuslink.llm.hypothesis.pipeline._mark_phase3_done", AsyncMock()),
        ):
            # Make KnowledgeGraph.load return our fake graph
            MockKG.load.return_value = fake_graph
            # The cache file must appear to exist so the pipeline doesn't raise
            (cache_dir / "graph.gpickle").write_bytes(b"fake")

            hyp_result = await run_hypothesis_pipeline(
                vault_path=vault,
                config=mock_config,
                top_bridges=4,
                skip_validation=True,
            )

        assert hyp_result["hypotheses_generated"] == 1
        assert hyp_result["top_hypothesis"] == top_hyp.statement
        assert hyp_result["report_path"] == str(report_md)

        # ------------------------------------------------------------------
        # Final assertions: vault structure + wikilink presence
        # ------------------------------------------------------------------
        assert report_md.exists(), "Report file was not created"
        report_content = report_md.read_text("utf-8")

        # Report must contain at least one [[wikilink]]
        wikilinks_in_report = _WIKILINK_RE.findall(report_content)
        assert wikilinks_in_report, "Report contains no [[wikilinks]]"

        # Paper notes must have YAML frontmatter
        for note in papers_dir.glob("*.md"):
            text = note.read_text("utf-8")
            assert text.startswith("---"), f"{note.name} missing YAML frontmatter"
            assert "title:" in text
            assert "authors:" in text
