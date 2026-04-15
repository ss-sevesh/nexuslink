"""Feedback loop — learns from human edits in Obsidian."""

import json
from pathlib import Path
from loguru import logger
from nexuslink.wiki.vault.reader import VaultReader
from nexuslink.wiki.vault.models import HypothesisNote, ConceptOverride


class FewShotSet:
    def __init__(self):
        self.positive_examples: list[str] = []
        self.negative_examples: list[str] = []
        self.calibration_notes: str = ""


class ScoringCalibration:
    def __init__(self):
        self.novelty_bias: float = 0.0
        self.feasibility_bias: float = 0.0
        self.impact_bias: float = 0.0
        self.mechanistic_depth_bias: float = 0.0
        self.falsifiability_bias: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringCalibration":
        sc = cls()
        for k, v in d.items():
            if hasattr(sc, k):
                setattr(sc, k, v)
        return sc


class FeedbackLoop:
    def __init__(self, reader: VaultReader):
        self.reader = reader
        self.cache_dir = reader.vault_path / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_path = self.cache_dir / "scoring_calibration.json"

    def collect_reviewed_hypotheses(self) -> list[HypothesisNote]:
        return [h for h in self.reader.read_all_hypotheses() if h.status in ("reviewed", "validated")]

    def collect_rejected_hypotheses(self) -> list[HypothesisNote]:
        return [h for h in self.reader.read_all_hypotheses() if h.status == "rejected"]

    def build_few_shot_examples(self, n_positive: int = 3, n_negative: int = 2) -> FewShotSet:
        fs = FewShotSet()

        reviewed = self.collect_reviewed_hypotheses()
        reviewed_edited = [h for h in reviewed if h.human_edited]
        if not reviewed_edited:
            reviewed_edited = reviewed
        reviewed_edited.sort(key=lambda h: h.composite_score, reverse=True)

        for h in reviewed_edited[:n_positive]:
            fs.positive_examples.append(
                f"GOOD HYPOTHESIS (score={h.composite_score}):\n{h.statement}\nDomains: {h.domains_spanned}"
            )

        rejected = self.collect_rejected_hypotheses()
        for h in rejected[:n_negative]:
            fs.negative_examples.append(
                f"BAD HYPOTHESIS (rejected):\n{h.statement}\nDomains: {h.domains_spanned}"
            )

        if fs.positive_examples or fs.negative_examples:
            fs.calibration_notes = f"Based on {len(fs.positive_examples)} reviewed and {len(fs.negative_examples)} rejected hypotheses from human feedback."
        else:
            fs.calibration_notes = "No human feedback available yet."

        logger.info(f"Built few-shot set: {len(fs.positive_examples)} positive, {len(fs.negative_examples)} negative")
        return fs

    def calibrate_scoring(self) -> ScoringCalibration:
        cal = ScoringCalibration()
        reviewed = [h for h in self.collect_reviewed_hypotheses() if h.human_edited]
        if not reviewed:
            logger.info("No human-edited scores found, skipping calibration")
            return cal

        # Load original LLM scores from cycle history if available
        cycle_history_path = self.cache_dir / "cycle_history.json"
        original_scores = {}
        if cycle_history_path.exists():
            try:
                history = json.loads(cycle_history_path.read_text())
                for entry in history:
                    if "hypothesis_scores" in entry:
                        original_scores.update(entry["hypothesis_scores"])
            except Exception:
                pass

        if not original_scores:
            logger.info("No original LLM scores to compare against")
            return cal

        diffs = {"novelty": [], "feasibility": [], "impact": [], "mechanistic_depth": [], "falsifiability": []}
        for h in reviewed:
            if h.id in original_scores:
                orig = original_scores[h.id]
                diffs["novelty"].append(h.novelty_score - orig.get("novelty_score", h.novelty_score))
                diffs["feasibility"].append(h.feasibility_score - orig.get("feasibility_score", h.feasibility_score))
                diffs["impact"].append(h.impact_score - orig.get("impact_score", h.impact_score))
                diffs["mechanistic_depth"].append(h.mechanistic_depth - orig.get("mechanistic_depth", h.mechanistic_depth))
                diffs["falsifiability"].append(h.falsifiability_score - orig.get("falsifiability_score", h.falsifiability_score))

        for key, vals in diffs.items():
            if vals:
                setattr(cal, f"{key}_bias", sum(vals) / len(vals))

        self.calibration_path.write_text(json.dumps(cal.to_dict(), indent=2))
        logger.info(f"Scoring calibration saved: {cal.to_dict()}")
        return cal

    def load_calibration(self) -> ScoringCalibration:
        if self.calibration_path.exists():
            return ScoringCalibration.from_dict(json.loads(self.calibration_path.read_text()))
        return ScoringCalibration()

    def get_human_concept_overrides(self) -> dict[str, ConceptOverride]:
        overrides = {}
        for concept in self.reader.read_all_concepts():
            if concept.human_edited:
                overrides[concept.name] = ConceptOverride(
                    concept_name=concept.name,
                    human_type=concept.entity_type,
                    human_domains=concept.domains,
                    additional_context="",
                )
        return overrides

    def get_rejected_bridge_pairs(self) -> set[tuple[str, str]]:
        """Return bridge pairs that produced rejected hypotheses — skip these next cycle."""
        rejected = self.collect_rejected_hypotheses()
        pairs = set()
        for h in rejected:
            content = h.raw_content
            import re
            bridges = re.findall(r'\[\[(.+?)\]\]\s*↔\s*\[\[(.+?)\]\]', content)
            for a, b in bridges:
                pairs.add((min(a, b), max(a, b)))
        return pairs
