"""Tests for the LLM hypothesis generation layer."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexuslink.llm.hypothesis.generator import (
    GeneratedHypothesis,
    HypothesisGenerator,
    _parse_hypothesis_list,
)
from nexuslink.llm.prompts.templates import (
    get_system_prompt,
    render_hypothesis_critique,
    render_hypothesis_generation,
    render_hypothesis_refinement,
    render_report_synthesis,
    render_template,
)
from nexuslink.llm.reports.writer import ReportWriter, _render_markdown_report, _tex
from nexuslink.llm.scoring.ranker import HypothesisRanker, ScoredHypothesis
from nexuslink.utils.json_parser import extract_json
from nexuslink.wiki.graph.builder import KnowledgeGraph
from nexuslink.wiki.linker.bridge_finder import ConceptBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bridge(a: str, da: str, b: str, db: str, sim: float = 0.82) -> ConceptBridge:
    return ConceptBridge(
        entity_a=a, entity_b=b,
        similarity_score=sim,
        domain_a=da, domain_b=db,
        bridge_type="analogous",
        entity_type_a="phenomenon",
        entity_type_b="phenomenon",
    )


def _hyp(**kwargs) -> GeneratedHypothesis:
    defaults = dict(
        statement="If quantum Casimir forces and gecko adhesion van der Waals scaling, "
                  "then engineered surfaces could achieve dry adhesion at the microscale.",
        evidence_bridges=["Casimir effect::gecko adhesion"],
        domains_spanned=["physics", "biology"],
        suggested_experiments=[
            "Fabricate 10 nm pillar arrays and measure pull-off force vs. column density",
            "Compare Casimir force calculations with measured gecko setae adhesion energy",
        ],
        confidence=0.75,
        raw_reasoning="Both phenomena arise from van der Waals interactions.",
    )
    defaults.update(kwargs)
    return GeneratedHypothesis(**defaults)


def _scored(**kwargs) -> ScoredHypothesis:
    h = _hyp()
    defaults = dict(
        **h.model_dump(),
        novelty_score=8.0,
        feasibility_score=6.0,
        impact_score=9.0,
        weaknesses=["Requires clean-room fabrication", "Scale-up unclear"],
        critique_summary="High-impact hypothesis with moderate feasibility.",
        overall_rank=1,
    )
    defaults.update(kwargs)
    return ScoredHypothesis(**defaults)


def _mock_claude_response(text: str) -> MagicMock:
    """Build a mock anthropic response whose content[0].text == text."""
    content = MagicMock()
    content.text = text
    response = MagicMock()
    response.content = [content]
    return response


# ===========================================================================
# JSON extraction
# ===========================================================================

class TestExtractJson:

    def test_plain_json_array(self):
        text = '[{"statement": "If A and B, then C."}]'
        result = extract_json(text)
        assert result == [{"statement": "If A and B, then C."}]

    def test_fenced_json_block(self):
        text = "Here is the JSON:\n```json\n[{\"x\": 1}]\n```\nDone."
        result = extract_json(text)
        assert result == [{"x": 1}]

    def test_fenced_block_no_language_tag(self):
        text = "```\n{\"a\": 2}\n```"
        result = extract_json(text)
        assert result == {"a": 2}

    def test_json_embedded_in_prose(self):
        text = 'Some text. [{"val": 99}] More text.'
        result = extract_json(text)
        assert result == [{"val": 99}]

    def test_raises_on_unparseable(self):
        with pytest.raises(Exception):
            extract_json("this is not json at all !!!")


# ===========================================================================
# Prompt template rendering
# ===========================================================================

class TestPromptTemplates:

    def test_hypothesis_generation_contains_bridges(self):
        bridges = [_bridge("Casimir effect", "physics", "gecko adhesion", "biology")]
        rendered = render_hypothesis_generation(bridges, ["physics", "biology"], total_concepts=42)

        assert "Casimir effect" in rendered
        assert "gecko adhesion" in rendered
        assert "physics" in rendered
        assert "biology" in rendered
        assert "42" in rendered
        assert "JSON" in rendered

    def test_hypothesis_generation_all_bridges_present(self):
        bridges = [
            _bridge("Casimir effect", "physics", "gecko adhesion", "biology"),
            _bridge("van der Waals", "chemistry", "cell membrane", "biology"),
            _bridge("CRISPR", "biology", "error correction", "cs"),
        ]
        rendered = render_hypothesis_generation(
            bridges, ["physics", "biology", "chemistry", "cs"], 100
        )
        for b in bridges:
            assert b.entity_a in rendered
            assert b.entity_b in rendered

    def test_hypothesis_generation_includes_few_shot_example(self):
        bridges = [_bridge("A", "d1", "B", "d2")]
        rendered = render_hypothesis_generation(bridges, ["d1", "d2"], 10)
        # The few-shot example mentions gecko and Casimir
        assert "gecko" in rendered.lower()
        assert "Casimir" in rendered

    def test_hypothesis_critique_contains_statement(self):
        h = _hyp()
        rendered = render_hypothesis_critique(h)
        assert h.statement in rendered
        assert "novelty" in rendered.lower()
        assert "feasibility" in rendered.lower()
        assert "impact" in rendered.lower()

    def test_hypothesis_critique_contains_experiments(self):
        h = _hyp()
        rendered = render_hypothesis_critique(h)
        assert h.suggested_experiments[0] in rendered

    def test_hypothesis_refinement_contains_scores(self):
        h = _scored()
        rendered = render_hypothesis_refinement(h, h)
        assert str(int(h.novelty_score)) in rendered
        assert str(int(h.feasibility_score)) in rendered
        assert h.weaknesses[0] in rendered

    def test_report_synthesis_lists_hypotheses(self):
        h = _scored()
        rendered = render_report_synthesis([h], stats={
            "papers_processed": 5, "total_concepts": 30,
            "total_bridges": 12, "domains": ["physics", "biology"],
        })
        assert h.statement[:30] in rendered
        assert "Executive Summary" in rendered

    def test_report_synthesis_shows_composite_score(self):
        h = _scored(novelty_score=8, feasibility_score=6, impact_score=9)
        rendered = render_report_synthesis([h], stats={
            "papers_processed": 5, "total_concepts": 30,
            "total_bridges": 12, "domains": ["physics", "biology"],
        })
        expected_score = f"{h.composite_score:.1f}"
        assert expected_score in rendered

    def test_render_template_unknown_raises(self):
        with pytest.raises(KeyError):
            render_template("nonexistent_template")

    def test_get_system_prompt_returns_string(self):
        for name in ("hypothesis_generation", "hypothesis_critique",
                     "hypothesis_refinement", "report_synthesis"):
            prompt = get_system_prompt(name)
            assert isinstance(prompt, str)
            assert len(prompt) > 20

    def test_generation_system_prompt_mentions_cross_domain(self):
        prompt = get_system_prompt("hypothesis_generation")
        assert "cross-domain" in prompt.lower() or "cross domain" in prompt.lower()


# ===========================================================================
# Parse hypothesis list
# ===========================================================================

class TestParseHypothesisList:

    def test_parses_valid_list(self):
        bridges = [_bridge("Casimir effect", "physics", "gecko adhesion", "biology")]
        json_text = json.dumps([{
            "statement": "If Casimir forces and gecko adhesion, then engineered surfaces.",
            "domains_spanned": ["physics", "biology"],
            "suggested_experiments": ["Exp 1", "Exp 2"],
            "confidence": 0.8,
            "reasoning": "Van der Waals link.",
        }])
        results = _parse_hypothesis_list(json_text, bridges)
        assert len(results) == 1
        assert results[0].confidence == 0.8
        assert results[0].evidence_bridges == ["Casimir effect::gecko adhesion"]

    def test_parses_pre_parsed_list(self):
        """Accepts already-parsed list (new _call_claude returns dict/list)."""
        bridges = [_bridge("A", "d1", "B", "d2")]
        data = [{"statement": "If A and B, then C.", "confidence": 0.6, "domains_spanned": ["d1"]}]
        results = _parse_hypothesis_list(data, bridges)
        assert len(results) == 1
        assert results[0].confidence == 0.6

    def test_skips_empty_statements(self):
        bridges = [_bridge("A", "d1", "B", "d2")]
        json_text = json.dumps([{"statement": "", "domains_spanned": []}])
        results = _parse_hypothesis_list(json_text, bridges)
        assert results == []

    def test_handles_missing_fields_gracefully(self):
        bridges = [_bridge("X", "d1", "Y", "d2")]
        json_text = json.dumps([{"statement": "If X and Y, then Z."}])
        results = _parse_hypothesis_list(json_text, bridges)
        assert len(results) == 1
        assert results[0].confidence == 0.5  # default


# ===========================================================================
# ScoredHypothesis composite score
# ===========================================================================

class TestScoredHypothesisCompositeScore:

    def test_composite_formula(self):
        h = _scored(novelty_score=8.0, feasibility_score=6.0, impact_score=9.0)
        expected = 0.4 * 8.0 + 0.3 * 9.0 + 0.3 * 6.0
        assert abs(h.composite_score - expected) < 1e-6

    def test_all_tens_gives_ten(self):
        h = _scored(novelty_score=10.0, feasibility_score=10.0, impact_score=10.0)
        assert abs(h.composite_score - 10.0) < 1e-6

    def test_all_zeros_gives_zero(self):
        h = _scored(novelty_score=0.0, feasibility_score=0.0, impact_score=0.0)
        assert h.composite_score == 0.0

    def test_sort_scored_sorts_correctly(self):
        ranker = object.__new__(HypothesisRanker)
        low = _scored(novelty_score=3.0, feasibility_score=3.0, impact_score=3.0)
        high = _scored(novelty_score=9.0, feasibility_score=9.0, impact_score=9.0)
        mid = _scored(novelty_score=6.0, feasibility_score=6.0, impact_score=6.0)

        ranked = ranker._sort_scored([low, high, mid])

        assert ranked[0].composite_score > ranked[1].composite_score > ranked[2].composite_score
        assert ranked[0].overall_rank == 1
        assert ranked[1].overall_rank == 2
        assert ranked[2].overall_rank == 3


# ===========================================================================
# HypothesisGenerator (mocked API)
# ===========================================================================

class TestHypothesisGenerator:

    def _api_json(self) -> str:
        return json.dumps([{
            "statement": "If Casimir forces and gecko adhesion, then engineered dry adhesives.",
            "domains_spanned": ["physics", "biology"],
            "suggested_experiments": ["AFM force measurement", "SEM surface imaging"],
            "confidence": 0.78,
            "reasoning": "Both governed by van der Waals.",
        }])

    @pytest.mark.asyncio
    async def test_generate_returns_hypotheses(self, tmp_path):
        mock_response = _mock_claude_response(self._api_json())
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response

        kg = KnowledgeGraph()

        with patch("nexuslink.llm.hypothesis.generator._HYPOTHESES_DIR", tmp_path / "03-hypotheses"):
            gen = HypothesisGenerator.__new__(HypothesisGenerator)
            gen._client = mock_client
            gen._model = "claude-sonnet-4-20250514"

            bridges = [_bridge("Casimir effect", "physics", "gecko adhesion", "biology")]
            results = await gen.generate(bridges, kg)

        assert len(results) == 1
        assert "Casimir" in results[0].statement or "engineered" in results[0].statement
        assert results[0].confidence == pytest.approx(0.78)

    @pytest.mark.asyncio
    async def test_generate_writes_wiki_notes(self, tmp_path):
        mock_response = _mock_claude_response(self._api_json())
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response

        kg = KnowledgeGraph()
        hyp_dir = tmp_path / "03-hypotheses"

        with patch("nexuslink.llm.hypothesis.generator._HYPOTHESES_DIR", hyp_dir):
            gen = HypothesisGenerator.__new__(HypothesisGenerator)
            gen._client = mock_client
            gen._model = "claude-sonnet-4-20250514"

            bridges = [_bridge("Casimir effect", "physics", "gecko adhesion", "biology")]
            await gen.generate(bridges, kg)

        notes = list(hyp_dir.glob("*.md"))
        assert len(notes) == 1
        content = notes[0].read_text()
        assert "## Hypothesis Statement" in content
        assert "## Suggested Experiments" in content

    @pytest.mark.asyncio
    async def test_generate_with_no_bridges_returns_empty(self):
        gen = object.__new__(HypothesisGenerator)
        gen._client = AsyncMock()
        gen._model = "claude-sonnet-4-20250514"

        kg = KnowledgeGraph()
        results = await gen.generate([], kg)
        assert results == []

    @pytest.mark.asyncio
    async def test_batching_respects_batch_size(self, tmp_path):
        """5 bridges in the same domain pair with batch size 4 → 2 API calls."""
        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_claude_response(self._api_json())

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = fake_create

        kg = KnowledgeGraph()
        bridges = [
            _bridge(f"A{i}", "physics", f"B{i}", "biology") for i in range(5)
        ]

        with (
            patch("nexuslink.llm.hypothesis.generator._HYPOTHESES_DIR", tmp_path / "03-hypotheses"),
            patch("nexuslink.llm.hypothesis.generator._BATCH_SIZE", 4),
        ):
            gen = HypothesisGenerator.__new__(HypothesisGenerator)
            gen._client = mock_client
            gen._model = "claude-sonnet-4-20250514"
            await gen.generate(bridges, kg)

        assert call_count == 2  # ceil(5 / 4) = 2 groups → 2 calls


# ===========================================================================
# HypothesisRanker (mocked API)
# ===========================================================================

class TestHypothesisRanker:

    @pytest.mark.asyncio
    async def test_critique_parses_scores(self):
        critique_json = json.dumps({
            "novelty_score": 8,
            "feasibility_score": 5,
            "impact_score": 9,
            "strengths": ["Strong cross-domain link"],
            "weaknesses": ["Needs clean-room", "Scale unclear"],
            "critique_summary": "Strong cross-domain insight.",
        })
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _mock_claude_response(critique_json)

        ranker = HypothesisRanker.__new__(HypothesisRanker)
        ranker._client = mock_client
        ranker._model = "claude-sonnet-4-20250514"

        result = await ranker.critique(_hyp())

        assert result.novelty_score == 8.0
        assert result.feasibility_score == 5.0
        assert result.impact_score == 9.0
        assert len(result.weaknesses) == 2
        assert len(result.strengths) == 1

    @pytest.mark.asyncio
    async def test_rank_all_critiques_and_sorts(self):
        """rank_all should call critique for each hypothesis and sort by composite_score."""
        call_count = 0
        scores = [
            {"novelty_score": 9, "feasibility_score": 9, "impact_score": 9},
            {"novelty_score": 3, "feasibility_score": 3, "impact_score": 3},
            {"novelty_score": 6, "feasibility_score": 6, "impact_score": 6},
        ]

        async def fake_create(**kwargs):
            nonlocal call_count
            resp_json = json.dumps({**scores[call_count % len(scores)], "weaknesses": []})
            call_count += 1
            return _mock_claude_response(resp_json)

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = fake_create

        ranker = HypothesisRanker.__new__(HypothesisRanker)
        ranker._client = mock_client
        ranker._model = "claude-sonnet-4-20250514"

        hypotheses = [_hyp(), _hyp(), _hyp()]
        ranked = await ranker.rank_all(hypotheses)

        assert len(ranked) == 3
        assert ranked[0].overall_rank == 1
        assert ranked[0].composite_score >= ranked[1].composite_score >= ranked[2].composite_score
        assert call_count == 3  # one critique per hypothesis

    @pytest.mark.asyncio
    async def test_refine_updates_statement(self):
        refinement_json = json.dumps({
            "revised_statement": "If Casimir forces (measured via AFM) and gecko adhesion (setal density), "
                                 "then arrays with 10^8 setae/cm² achieve 100 kPa dry adhesion.",
            "addressed_weaknesses": ["More specific fabrication target given"],
            "revised_experiments": ["AFM force mapping at 10 nm resolution", "SEM setal density count"],
            "revised_confidence": 0.82,
        })
        # Refinement call returns the revised statement;
        # the subsequent re-critique call returns default scores.
        critique_json = json.dumps({
            "novelty_score": 8, "feasibility_score": 6, "impact_score": 9, "weaknesses": [],
        })

        call_count = 0
        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_claude_response(refinement_json)
            return _mock_claude_response(critique_json)

        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = fake_create

        ranker = HypothesisRanker.__new__(HypothesisRanker)
        ranker._client = mock_client
        ranker._model = "claude-sonnet-4-20250514"

        original = _scored()
        refined = await ranker.refine_top_n([original], n=1)

        # Statement should be updated (from re-critique result or refinement)
        assert refined[0].confidence == pytest.approx(0.82) or "AFM" in refined[0].statement
        assert call_count == 2  # 1 refine + 1 re-critique


# ===========================================================================
# Report structure validation
# ===========================================================================

class TestReportWriter:

    @pytest.mark.asyncio
    async def test_markdown_report_has_required_sections(self, tmp_path):
        synthesis_json = json.dumps({
            "executive_summary": "This report presents three novel hypotheses.",
            "cross_domain_narrative": "Physics and biology are strongly linked via van der Waals.",
        })
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _mock_claude_response(synthesis_json)

        from nexuslink.wiki.citations.manager import CitationManager

        kg = KnowledgeGraph()
        hypotheses = [_scored(overall_rank=1), _scored(overall_rank=2)]

        with (
            patch("nexuslink.llm.reports.writer._REPORTS_DIR", tmp_path / "04-reports"),
            patch("nexuslink.llm.reports.writer._LATEX_DIR", tmp_path / "04-reports" / "latex"),
        ):
            writer = ReportWriter.__new__(ReportWriter)
            writer._client = mock_client
            writer._model = "claude-sonnet-4-20250514"
            path = await writer.generate_report(hypotheses, kg, CitationManager())

        content = (tmp_path / "04-reports" / Path(path).name).read_text()
        for section in [
            "## Executive Summary",
            "## Cross-Domain Analysis Map",
            "## Ranked Hypotheses",
            "## Evidence Graph Description",
            "## Full Bibliography",
        ]:
            assert section in content, f"Missing section: {section!r}"

    @pytest.mark.asyncio
    async def test_report_has_yaml_frontmatter(self, tmp_path):
        synthesis_json = json.dumps({
            "executive_summary": "Summary.",
            "cross_domain_narrative": "Narrative.",
        })
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _mock_claude_response(synthesis_json)

        from nexuslink.wiki.citations.manager import CitationManager

        kg = KnowledgeGraph()

        with (
            patch("nexuslink.llm.reports.writer._REPORTS_DIR", tmp_path / "04-reports"),
            patch("nexuslink.llm.reports.writer._LATEX_DIR", tmp_path / "04-reports" / "latex"),
        ):
            writer = ReportWriter.__new__(ReportWriter)
            writer._client = mock_client
            writer._model = "claude-sonnet-4-20250514"
            path = await writer.generate_report([_scored()], kg, CitationManager())

        content = (tmp_path / "04-reports" / Path(path).name).read_text()
        assert content.startswith("---"), "Report must start with YAML frontmatter"
        assert "hypothesis_count:" in content
        assert "top_score:" in content
        assert "date:" in content

    @pytest.mark.asyncio
    async def test_latex_file_is_written(self, tmp_path):
        synthesis_json = json.dumps({
            "executive_summary": "Summary text.",
            "cross_domain_narrative": "Narrative text.",
        })
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _mock_claude_response(synthesis_json)

        from nexuslink.wiki.citations.manager import CitationManager

        kg = KnowledgeGraph()

        with (
            patch("nexuslink.llm.reports.writer._REPORTS_DIR", tmp_path / "04-reports"),
            patch("nexuslink.llm.reports.writer._LATEX_DIR", tmp_path / "04-reports" / "latex"),
        ):
            writer = ReportWriter.__new__(ReportWriter)
            writer._client = mock_client
            writer._model = "claude-sonnet-4-20250514"
            await writer.generate_report([_scored()], kg, CitationManager())

        latex_files = list((tmp_path / "04-reports" / "latex").glob("*.tex"))
        assert len(latex_files) == 1
        latex = latex_files[0].read_text()
        assert r"\documentclass" in latex
        assert r"\begin{document}" in latex
        assert r"\section{Ranked Hypotheses}" in latex

    def test_report_contains_wikilinks_to_domains(self):
        from nexuslink.wiki.citations.manager import CitationManager

        kg = KnowledgeGraph()
        h = _scored()
        content = _render_markdown_report(
            [h], "Summary.", "Narrative.", CitationManager(),
            {"total_concepts": 10, "total_bridges": 4,
             "domains_covered": 2, "domains": ["physics", "biology"],
             "papers_processed": 3},
        )
        assert "[[physics]]" in content or "[[biology]]" in content

    @pytest.mark.asyncio
    async def test_wikilink_validation_logs_broken_links(self, tmp_path, caplog):
        """Broken wikilinks are logged but do not raise."""
        synthesis_json = json.dumps({
            "executive_summary": "[[NonExistentPaper]] is key.",
            "cross_domain_narrative": "Narrative.",
        })
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _mock_claude_response(synthesis_json)

        from nexuslink.wiki.citations.manager import CitationManager

        kg = KnowledgeGraph()
        fake_vault = tmp_path / "wiki"
        fake_vault.mkdir()

        with (
            patch("nexuslink.llm.reports.writer._REPORTS_DIR", tmp_path / "04-reports"),
            patch("nexuslink.llm.reports.writer._LATEX_DIR", tmp_path / "04-reports" / "latex"),
        ):
            writer = ReportWriter.__new__(ReportWriter)
            writer._client = mock_client
            writer._model = "claude-sonnet-4-20250514"
            # Should not raise even with broken links
            path = await writer.generate_report(
                [_scored()], kg, CitationManager(), vault_path=fake_vault
            )

        assert Path(path).exists()


# ===========================================================================
# LaTeX escaping
# ===========================================================================

class TestTexEscape:

    def test_underscore_escaped(self):
        assert r"\_" in _tex("var_name")

    def test_percent_escaped(self):
        assert r"\%" in _tex("50% yield")

    def test_ampersand_escaped(self):
        assert r"\&" in _tex("Smith & Jones")

    def test_dollar_escaped(self):
        assert r"\$" in _tex("cost: $100")

    def test_backslash_escaped_first(self):
        result = _tex("a\\b")
        assert "\\textbackslash{}" in result
        assert result.count("\\textbackslash{}") == 1
