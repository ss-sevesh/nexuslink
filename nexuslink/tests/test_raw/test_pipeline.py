"""Tests for the RAW ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexuslink.raw.ingestion.pipeline import _detect_source_type, _render_paper_note, run_ingestion
from nexuslink.raw.ingestion.pdf_loader import _extract_authors, _extract_title, ingest_pdf
from nexuslink.raw.schemas.models import ExtractedEntity, RawDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(**kwargs) -> RawDocument:
    defaults = dict(title="Test Paper", authors=["Alice Smith", "Bob Jones"],
                    full_text="Some text.", source_path="test.pdf")
    defaults.update(kwargs)
    return RawDocument(**defaults)


# ---------------------------------------------------------------------------
# Source type detection
# ---------------------------------------------------------------------------

class TestDetectSourceType:
    def test_pdf_by_extension(self, tmp_path):
        p = tmp_path / "paper.pdf"
        p.write_bytes(b"%PDF")
        assert _detect_source_type(str(p)) == "pdf"

    def test_pdf_extension_without_existing_file(self):
        assert _detect_source_type("/nonexistent/path/paper.pdf") == "pdf"

    def test_arxiv_id_simple(self):
        assert _detect_source_type("2101.03961") == "arxiv"

    def test_arxiv_id_with_version(self):
        assert _detect_source_type("2101.03961v2") == "arxiv"

    def test_arxiv_id_five_digits(self):
        assert _detect_source_type("2310.12345") == "arxiv"

    def test_bare_doi(self):
        assert _detect_source_type("10.1038/nature12345") == "doi"

    def test_doi_with_subdirectory(self):
        assert _detect_source_type("10.1016/j.cell.2021.01.001") == "doi"

    def test_doi_org_url(self):
        assert _detect_source_type("https://doi.org/10.1038/nature12345") == "doi"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot detect source type"):
            _detect_source_type("not-an-id-or-path")


# ---------------------------------------------------------------------------
# PDF title / author heuristics
# ---------------------------------------------------------------------------

class TestPdfHeuristics:
    def test_extract_title_from_heading(self):
        md = "# Attention Is All You Need\n\nAuthors...\n\nAbstract..."
        assert _extract_title(md, Path("paper.pdf")) == "Attention Is All You Need"

    def test_extract_title_from_h2_fallback(self):
        md = "## A Second-Level Title\n\nSome text."
        assert _extract_title(md, Path("paper.pdf")) == "A Second-Level Title"

    def test_extract_title_fallback_to_stem(self):
        md = "http://example.com\nhttp://other.com"
        assert _extract_title(md, Path("my_paper.pdf")) == "my_paper"

    def test_extract_authors_comma_separated(self):
        md = "# Title\nAlice Smith, Bob Jones, Carol Lee\n\nAbstract here."
        authors = _extract_authors(md)
        assert authors == ["Alice Smith", "Bob Jones", "Carol Lee"]

    def test_extract_authors_with_and(self):
        md = "# Title\nAlice Smith, Bob Jones and Carol Lee\n\nAbstract."
        authors = _extract_authors(md)
        assert "Alice Smith" in authors
        assert "Carol Lee" in authors

    def test_extract_authors_skips_abstract(self):
        md = "# Title\nAbstract: This paper presents...\nAlice Smith, Bob Jones"
        authors = _extract_authors(md)
        # The abstract line should be skipped; authors line found
        assert "Alice Smith" in authors

    def test_extract_authors_empty_when_no_match(self):
        md = "# Title\nNo names here at all.\n\nJust paragraphs."
        assert _extract_authors(md) == []


# ---------------------------------------------------------------------------
# PDF ingestion (mocked pymupdf4llm)
# ---------------------------------------------------------------------------

class TestIngestPdf:
    @pytest.mark.asyncio
    async def test_returns_raw_document(self, tmp_path):
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        markdown = "# Attention Is All You Need\nVaswani, Shazeer, Parmar\n\nThis paper proposes..."

        with patch("nexuslink.raw.ingestion.pdf_loader.pymupdf4llm.to_markdown",
                   return_value=markdown):
            doc = await ingest_pdf(pdf_path)

        assert doc.title == "Attention Is All You Need"
        assert "Vaswani" in doc.authors[0]
        assert doc.full_text == markdown
        assert doc.source_path == str(pdf_path)
        assert doc.id  # non-empty UUID

    @pytest.mark.asyncio
    async def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await ingest_pdf(tmp_path / "nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_title_fallback_to_stem(self, tmp_path):
        pdf_path = tmp_path / "my_paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("nexuslink.raw.ingestion.pdf_loader.pymupdf4llm.to_markdown",
                   return_value="http://example.com\nsome noise"):
            doc = await ingest_pdf(pdf_path)

        assert doc.title == "my_paper"


# ---------------------------------------------------------------------------
# Entity extraction (mocked spaCy)
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def _make_mock_nlp(self, ents: list[tuple[str, str, str]]) -> MagicMock:
        """Build a mock nlp where *ents* is [(text, label_, sent_text), ...]."""
        mock_ents = []
        for text, label, sent_text in ents:
            ent = MagicMock()
            ent.text = text
            ent.label_ = label
            ent.sent.text = sent_text
            mock_ents.append(ent)

        spacy_doc = MagicMock()
        spacy_doc.ents = mock_ents

        # PhraseMatcher returns no phrase matches in the mock path
        with patch("nexuslink.raw.extraction.entity_extractor._build_phrase_matcher") as mock_pm:
            mock_pm.return_value = MagicMock(return_value=[])

        mock_nlp = MagicMock(return_value=spacy_doc)
        return mock_nlp

    def test_chemical_entity_extracted(self):
        from nexuslink.raw.extraction.entity_extractor import extract_entities

        doc = _make_doc(full_text="We treated cells with Taxol, a well-known CHEMICAL.")
        mock_ent = MagicMock()
        mock_ent.text = "Taxol"
        mock_ent.label_ = "CHEMICAL"
        mock_ent.sent.text = "We treated cells with Taxol, a well-known CHEMICAL."

        spacy_doc = MagicMock()
        spacy_doc.ents = [mock_ent]

        with patch("nexuslink.raw.extraction.entity_extractor._build_phrase_matcher") as mock_pm_fn:
            mock_pm_fn.return_value = MagicMock(return_value=[])
            entities = extract_entities(doc, nlp=MagicMock(return_value=spacy_doc))

        assert len(entities) == 1
        assert entities[0].name == "Taxol"
        assert entities[0].entity_type == "chemical"
        assert entities[0].source_doc_id == doc.id

    def test_unknown_label_skipped(self):
        from nexuslink.raw.extraction.entity_extractor import extract_entities

        doc = _make_doc(full_text="This happened in 2021.")
        mock_ent = MagicMock()
        mock_ent.text = "2021"
        mock_ent.label_ = "DATE"
        mock_ent.sent.text = "This happened in 2021."

        spacy_doc = MagicMock()
        spacy_doc.ents = [mock_ent]

        with patch("nexuslink.raw.extraction.entity_extractor._build_phrase_matcher") as mock_pm_fn:
            mock_pm_fn.return_value = MagicMock(return_value=[])
            entities = extract_entities(doc, nlp=MagicMock(return_value=spacy_doc))

        assert entities == []

    def test_deduplication(self):
        from nexuslink.raw.extraction.entity_extractor import extract_entities

        doc = _make_doc(full_text="CRISPR was used. CRISPR is powerful.")
        # Two ents with same text and label
        def make_ent(sent_text):
            e = MagicMock()
            e.text = "CRISPR"
            e.label_ = "method"  # not in _LABEL_MAP, so it won't come via NER path
            e.sent.text = sent_text
            return e

        spacy_doc = MagicMock()
        spacy_doc.ents = [make_ent("CRISPR was used."), make_ent("CRISPR is powerful.")]

        with patch("nexuslink.raw.extraction.entity_extractor._build_phrase_matcher") as mock_pm_fn:
            mock_pm_fn.return_value = MagicMock(return_value=[])
            entities = extract_entities(doc, nlp=MagicMock(return_value=spacy_doc))

        # Both have unknown label so neither is added via NER path; result is empty
        assert all(e.name != "CRISPR" for e in entities)


# ---------------------------------------------------------------------------
# Wiki note rendering
# ---------------------------------------------------------------------------

class TestRenderPaperNote:
    def test_frontmatter_fields_present(self):
        doc = _make_doc(title='My Paper: "With Quotes"', doi="10.1/test", year=2024,
                        domain_tags=["cs.AI"])
        entities: list[ExtractedEntity] = []
        note = _render_paper_note(doc, entities)

        assert 'title: "My Paper: \\"With Quotes\\""' in note
        assert 'doi: "10.1/test"' in note
        assert "year: 2024" in note
        assert '"cs.AI"' in note

    def test_entities_rendered_as_wikilinks(self):
        doc = _make_doc()
        entity = ExtractedEntity(
            name="CRISPR", entity_type="method",
            source_doc_id=doc.id, context_sentence="CRISPR was used."
        )
        note = _render_paper_note(doc, [entity])
        assert "[[CRISPR]]" in note

    def test_no_entities_placeholder(self):
        doc = _make_doc()
        note = _render_paper_note(doc, [])
        assert "<!-- none detected -->" in note

    def test_abstract_in_summary_section(self):
        doc = _make_doc(abstract="This paper studies X.")
        note = _render_paper_note(doc, [])
        assert "This paper studies X." in note


# ---------------------------------------------------------------------------
# Full pipeline (integration-level, all external I/O mocked)
# ---------------------------------------------------------------------------

class TestRunIngestion:
    @pytest.mark.asyncio
    async def test_pdf_pipeline_end_to_end(self, tmp_path):
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")
        markdown = "# Great Paper\nAlice Smith, Bob Jones\n\nWe used CRISPR."

        # Redirect wiki output to tmp_path so we don't pollute the repo
        wiki_papers = tmp_path / "wiki" / "papers"
        wiki_papers.mkdir(parents=True)

        with (
            patch("nexuslink.raw.ingestion.pdf_loader.pymupdf4llm.to_markdown",
                  return_value=markdown),
            patch("nexuslink.raw.ingestion.pipeline._WIKI_PAPERS_DIR", wiki_papers),
            patch("nexuslink.raw.extraction.entity_extractor._get_nlp") as mock_get_nlp,
        ):
            spacy_doc = MagicMock()
            spacy_doc.ents = []
            mock_nlp = MagicMock(return_value=spacy_doc)

            with patch("nexuslink.raw.extraction.entity_extractor._build_phrase_matcher") as mock_pm_fn:
                mock_pm_fn.return_value = MagicMock(return_value=[])
                mock_get_nlp.return_value = mock_nlp
                result = await run_ingestion(str(pdf_path))

        assert result["title"] == "Great Paper"
        assert result["source_type"] == "pdf"
        assert result["entities_found"] == 0
        assert (wiki_papers / "Great Paper.md").exists()

    @pytest.mark.asyncio
    async def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            await run_ingestion("completely-invalid-source")
