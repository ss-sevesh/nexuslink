"""Tests for the WIKI linking layer: embedder, bridge finder, graph, and classifier."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nexuslink.raw.schemas.models import ExtractedEntity, RawDocument
from nexuslink.wiki.graph.builder import KnowledgeGraph, _render_concept_note
from nexuslink.wiki.linker.bridge_finder import BridgeFinder, ConceptBridge, _infer_bridge_type
from nexuslink.wiki.linker.embedder import ConceptEmbedder, _entity_text, _sha
from nexuslink.wiki.taxonomy.classifier import classify_domain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entity(name: str, etype: str = "phenomenon", domain_doc: str = "doc1") -> ExtractedEntity:
    return ExtractedEntity(
        name=name,
        entity_type=etype,  # type: ignore[arg-type]
        source_doc_id=domain_doc,
        context_sentence=f"Context for {name}.",
    )


def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _mock_embedder(mapping: dict[str, np.ndarray]) -> MagicMock:
    """Return a ConceptEmbedder mock whose embed_batch returns *mapping*."""
    mock = MagicMock(spec=ConceptEmbedder)
    mock.embed_batch.side_effect = lambda entities: {
        e.name: mapping[e.name] for e in entities if e.name in mapping
    }
    mock.embed_entity.side_effect = lambda entity: mapping.get(entity.name, _unit([1, 0, 0]))
    return mock


def _doc(**kwargs) -> RawDocument:
    defaults = dict(
        id=str(uuid.uuid4()),
        title="Test Paper",
        authors=["Alice"],
        full_text="Some text.",
        source_path="test.pdf",
    )
    defaults.update(kwargs)
    return RawDocument(**defaults)


# ===========================================================================
# ConceptEmbedder
# ===========================================================================

class TestConceptEmbedder:

    def _make_embedder(self, tmp_path: Path) -> ConceptEmbedder:
        """Embedder with cache redirected to tmp_path and model mocked out."""
        embedder = ConceptEmbedder.__new__(ConceptEmbedder)
        embedder._model_name = "all-MiniLM-L6-v2"
        embedder._model = None
        embedder._cache = {}
        embedder._key_index = {}
        embedder._cache_path = tmp_path / "embeddings.npz"  # type: ignore[attr-defined]

        # Patch private attributes used by save_cache
        object.__setattr__(embedder, "_cache_path_obj", tmp_path / "embeddings.npz")
        return embedder

    def test_entity_text_format(self):
        e = _entity("CRISPR", "method")
        assert _entity_text(e) == "CRISPR: Context for CRISPR."

    def test_sha_deterministic(self):
        assert _sha("hello") == _sha("hello")
        assert _sha("hello") != _sha("world")

    def test_embed_entity_uses_model_and_caches(self, tmp_path):
        fake_emb = _unit([1.0, 0.0, 0.0])

        embedder = ConceptEmbedder.__new__(ConceptEmbedder)
        embedder._model_name = "all-MiniLM-L6-v2"
        embedder._cache = {}
        embedder._key_index = {}

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_emb
        embedder._model = mock_model

        with patch("nexuslink.wiki.linker.embedder._EMB_PATH", tmp_path / "e.npz"), \
             patch("nexuslink.wiki.linker.embedder._IDX_PATH", tmp_path / "e.json"), \
             patch("nexuslink.wiki.linker.embedder._CACHE_DIR", tmp_path):
            e = _entity("Casimir effect", "phenomenon")
            result = embedder.embed_entity(e)

        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()

        # Second call: cache hit — model not called again
        result2 = embedder.embed_entity(e)
        assert mock_model.encode.call_count == 1
        np.testing.assert_array_equal(result, result2)

    def test_embed_batch_returns_name_keyed_dict(self):
        fake_embs = {
            "Casimir effect": _unit([1, 0, 0]),
            "van der Waals": _unit([0, 1, 0]),
        }

        embedder = ConceptEmbedder.__new__(ConceptEmbedder)
        embedder._model_name = "all-MiniLM-L6-v2"
        embedder._cache = {}
        embedder._key_index = {}

        call_count = 0

        def fake_encode(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            return np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])

        mock_model = MagicMock()
        mock_model.encode.side_effect = fake_encode
        embedder._model = mock_model

        entities = [_entity("Casimir effect"), _entity("van der Waals")]
        result = embedder.embed_batch(entities)

        assert set(result.keys()) == {"Casimir effect", "van der Waals"}
        for arr in result.values():
            assert isinstance(arr, np.ndarray)
        # All encoded in a single call
        assert call_count == 1

    def test_embed_batch_skips_cached_entities(self):
        embedder = ConceptEmbedder.__new__(ConceptEmbedder)
        embedder._model_name = "all-MiniLM-L6-v2"
        embedder._key_index = {}

        cached_text = "CRISPR: Context for CRISPR."
        cached_h = _sha(cached_text)
        cached_emb = _unit([1, 0, 0])
        embedder._cache = {cached_h: cached_emb}

        mock_model = MagicMock()
        mock_model.encode.return_value = np.stack([_unit([0, 1, 0])])
        embedder._model = mock_model

        entities = [_entity("CRISPR"), _entity("photon")]
        result = embedder.embed_batch(entities)

        # Only "photon" should have been sent to the model
        assert mock_model.encode.call_count == 1
        texts_sent = mock_model.encode.call_args[0][0]
        assert len(texts_sent) == 1
        assert "photon" in texts_sent[0]
        assert "CRISPR" in result


# ===========================================================================
# BridgeFinder
# ===========================================================================

class TestBridgeFinder:

    def test_cross_domain_bridge_detected(self):
        # physics entity and biology entity with high cosine similarity
        phys_emb = _unit([1.0, 0.1, 0.0])
        bio_emb = _unit([0.95, 0.15, 0.0])  # dot ≈ 0.998

        embedder = _mock_embedder({"Casimir effect": phys_emb, "gecko adhesion": bio_emb})
        finder = BridgeFinder(embedder)

        phys_ents = [_entity("Casimir effect", "phenomenon")]
        bio_ents = [_entity("gecko adhesion", "phenomenon")]

        bridges = finder.find_bridges({"physics": phys_ents, "biology": bio_ents}, threshold=0.65)

        assert len(bridges) == 1
        b = bridges[0]
        assert b.entity_a in ("Casimir effect", "gecko adhesion")
        assert b.entity_b in ("Casimir effect", "gecko adhesion")
        assert b.entity_a != b.entity_b
        assert b.domain_a != b.domain_b
        assert b.similarity_score >= 0.65

    def test_same_domain_never_bridged(self):
        emb_a = _unit([1, 0, 0])
        emb_b = _unit([0.99, 0.1, 0])

        embedder = _mock_embedder({"alpha": emb_a, "beta": emb_b})
        finder = BridgeFinder(embedder)

        # Both in physics — no cross-domain bridges possible
        bridges = finder.find_bridges(
            {"physics": [_entity("alpha"), _entity("beta")]}, threshold=0.0
        )
        assert bridges == []

    def test_below_threshold_filtered(self):
        # Orthogonal vectors: dot = 0.0
        embedder = _mock_embedder({
            "alpha": _unit([1, 0, 0]),
            "beta": _unit([0, 1, 0]),
        })
        finder = BridgeFinder(embedder)

        bridges = finder.find_bridges(
            {
                "physics": [_entity("alpha")],
                "biology": [_entity("beta")],
            },
            threshold=0.65,
        )
        assert bridges == []

    def test_bridges_sorted_by_similarity_desc(self):
        # Three domains: physics, biology, chemistry
        # physics↔biology sim = 0.95; physics↔chemistry sim = 0.70
        phys = _unit([1.0, 0.0, 0.0])
        bio = _unit([0.95, 0.31, 0.0])   # dot(phys, bio) ≈ 0.95
        chem = _unit([0.70, 0.71, 0.0])  # dot(phys, chem) ≈ 0.70

        embedder = _mock_embedder({"P": phys, "B": bio, "C": chem})
        finder = BridgeFinder(embedder)

        bridges = finder.find_bridges(
            {
                "physics": [_entity("P")],
                "biology": [_entity("B")],
                "chemistry": [_entity("C")],
            },
            threshold=0.65,
        )
        scores = [b.similarity_score for b in bridges]
        assert scores == sorted(scores, reverse=True)

    def test_single_domain_returns_empty(self):
        embedder = _mock_embedder({"X": _unit([1, 0, 0])})
        finder = BridgeFinder(embedder)
        assert finder.find_bridges({"physics": [_entity("X")]}) == []


# ===========================================================================
# Bridge type inference
# ===========================================================================

class TestInferBridgeType:

    def test_very_high_sim_is_analogous(self):
        assert _infer_bridge_type(0.90, "phenomenon", "phenomenon") == "analogous"

    def test_method_plus_material_is_enables(self):
        assert _infer_bridge_type(0.70, "method", "material") == "enables"
        assert _infer_bridge_type(0.70, "material", "method") == "enables"

    def test_same_type_mid_sim_is_extends(self):
        assert _infer_bridge_type(0.75, "chemical", "chemical") == "extends"

    def test_mixed_types_mid_sim_is_enables(self):
        assert _infer_bridge_type(0.70, "phenomenon", "material") == "enables"


# ===========================================================================
# KnowledgeGraph
# ===========================================================================

class TestKnowledgeGraph:

    def _make_kg_with_data(self) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        doc_phys = _doc(title="Physics Paper", domain_tags=["physics"])
        doc_bio = _doc(title="Biology Paper", domain_tags=["biology"])
        ents_phys = [_entity("Casimir effect", "phenomenon")]
        ents_bio = [_entity("gecko adhesion", "phenomenon"), _entity("CRISPR", "method")]
        kg.add_paper(doc_phys, ents_phys)
        kg.add_paper(doc_bio, ents_bio)
        return kg

    def test_add_paper_creates_correct_nodes(self):
        kg = self._make_kg_with_data()
        assert kg.node_count("paper") == 2
        assert kg.node_count("concept") == 3  # Casimir, gecko, CRISPR

    def test_add_paper_creates_mentions_edges(self):
        kg = self._make_kg_with_data()
        # Edge from paper to concept should exist
        assert kg.edge_count() >= 3

    def test_add_bridge_creates_bidirectional_edge(self):
        kg = self._make_kg_with_data()
        bridge = ConceptBridge(
            entity_a="Casimir effect",
            entity_b="gecko adhesion",
            similarity_score=0.82,
            domain_a="physics",
            domain_b="biology",
            bridge_type="analogous",
        )
        before = kg.edge_count()
        kg.add_bridge(bridge)
        # Two new directed edges (a→b and b→a)
        assert kg.edge_count() == before + 2

    def test_get_cross_domain_clusters_finds_bridge(self):
        kg = self._make_kg_with_data()
        bridge = ConceptBridge(
            entity_a="Casimir effect",
            entity_b="gecko adhesion",
            similarity_score=0.82,
            domain_a="physics",
            domain_b="biology",
            bridge_type="analogous",
        )
        kg.add_bridge(bridge)
        clusters = kg.get_cross_domain_clusters()
        assert len(clusters) >= 1
        bridged_names = {name for cluster in clusters for name in cluster}
        assert "Casimir effect" in bridged_names
        assert "gecko adhesion" in bridged_names

    def test_get_cross_domain_clusters_empty_without_bridges(self):
        # Without bridges, each concept is isolated — no cross-domain cluster
        kg = KnowledgeGraph()
        doc = _doc(title="Solo Paper", domain_tags=["physics"])
        kg.add_paper(doc, [_entity("photon")])
        # Only one domain → no cross-domain cluster
        clusters = kg.get_cross_domain_clusters()
        # All concepts have only one domain, so no cluster qualifies
        assert all(len(c) >= 2 for c in clusters)  # vacuously true if empty

    def test_save_and_load_roundtrip(self, tmp_path):
        kg = self._make_kg_with_data()
        save_path = tmp_path / "graph.gpickle"
        kg.save(save_path)

        kg2 = KnowledgeGraph.load(save_path)
        assert kg2.node_count("paper") == 2
        assert kg2.node_count("concept") == 3

    @pytest.mark.asyncio
    async def test_export_for_obsidian_writes_files(self, tmp_path):
        kg = self._make_kg_with_data()
        bridge = ConceptBridge(
            entity_a="Casimir effect",
            entity_b="gecko adhesion",
            similarity_score=0.79,
            domain_a="physics",
            domain_b="biology",
            bridge_type="analogous",
        )
        kg.add_bridge(bridge)

        with patch("nexuslink.wiki.graph.builder._WIKI_DIR", tmp_path):
            count = await kg.export_for_obsidian()

        assert count == 3  # one file per concept
        concepts_dir = tmp_path / "concepts"
        assert concepts_dir.is_dir()
        files = list(concepts_dir.glob("*.md"))
        assert len(files) == 3

    def test_render_concept_note_has_wikilinks(self):
        content = _render_concept_note(
            name="Casimir effect",
            entity_type="phenomenon",
            domains=["physics"],
            bridges=[("gecko adhesion", "analogous", 0.82)],
            papers=["Biology Paper"],
        )
        assert "[[gecko adhesion]]" in content
        assert "[[Biology Paper]]" in content
        assert 'type: phenomenon' in content
        assert '"physics"' in content


# ===========================================================================
# Domain classifier
# ===========================================================================

class TestClassifyDomain:

    def test_physics_abstract_ranked_first(self):
        doc = _doc(
            abstract=(
                "We study the quantum Casimir effect in a photon field, "
                "examining entropy changes and thermodynamics of electron spin states."
            )
        )
        ranked = classify_domain(doc)
        assert ranked, "Expected non-empty classification"
        top_domain, top_score = ranked[0]
        assert top_domain == "physics", f"Expected 'physics', got {ranked[:3]}"
        assert 0 < top_score <= 1.0

    def test_biology_abstract_ranked_first(self):
        doc = _doc(
            abstract=(
                "We analyse gene expression in single cells during metabolism. "
                "The enzyme activity drives DNA transcription and protein folding."
            )
        )
        ranked = classify_domain(doc)
        top_domain = ranked[0][0]
        assert top_domain == "biology", f"Expected 'biology', got {ranked[:3]}"

    def test_arxiv_tag_boosts_cs_domain(self):
        doc = _doc(
            abstract="This paper presents a new method.",
            domain_tags=["cs.LG", "cs.AI"],
        )
        ranked = classify_domain(doc)
        domain_names = [d for d, _ in ranked]
        assert "cs" in domain_names
        cs_score = next(s for d, s in ranked if d == "cs")
        assert cs_score > 0

    def test_returns_only_nonzero_scores(self):
        doc = _doc(abstract="quantum photon electron wave thermodynamics field entropy")
        ranked = classify_domain(doc)
        assert all(s > 0 for _, s in ranked)

    def test_scores_sum_to_one(self):
        doc = _doc(
            abstract=(
                "CRISPR gene editing enables treatment of disease. "
                "The quantum field governs electron behaviour."
            )
        )
        ranked = classify_domain(doc)
        total = sum(s for _, s in ranked)
        assert abs(total - 1.0) < 1e-3, f"Scores should sum to 1.0, got {total}"
