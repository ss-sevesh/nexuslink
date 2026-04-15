from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime

class PaperNote(BaseModel):
    path: Path
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    doi: str = ""
    domain: str = ""
    year: int = 0
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    human_edited: bool = False
    last_pipeline_hash: str = ""
    raw_content: str = ""

class ConceptNote(BaseModel):
    path: Path
    name: str = ""
    entity_type: str = ""
    domains: list[str] = Field(default_factory=list)
    linked_concepts: list[str] = Field(default_factory=list)
    bridge_links: list[str] = Field(default_factory=list)
    paper_mentions: list[str] = Field(default_factory=list)
    human_edited: bool = False
    last_pipeline_hash: str = ""
    raw_content: str = ""

class HypothesisNote(BaseModel):
    path: Path
    id: str = ""
    status: str = "generated"  # generated|reviewed|validated|rejected
    confidence: float = 0.0
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    impact_score: float = 0.0
    mechanistic_depth: float = 0.0
    falsifiability_score: float = 0.0
    composite_score: float = 0.0
    domains_spanned: list[str] = Field(default_factory=list)
    cycle_generated: int = 0
    human_edited: bool = False
    last_pipeline_hash: str = ""
    statement: str = ""
    raw_content: str = ""

class VaultStats(BaseModel):
    total_papers: int = 0
    total_concepts: int = 0
    total_hypotheses: int = 0
    total_reports: int = 0
    total_cycles: int = 0
    hypotheses_by_status: dict[str, int] = Field(default_factory=dict)
    domains_covered: list[str] = Field(default_factory=list)
    broken_links: int = 0
    orphan_notes: int = 0

class HealReport(BaseModel):
    timestamp: str = ""
    duplicates_merged: int = 0
    links_fixed: int = 0
    concepts_pruned: int = 0
    notes_updated: int = 0

class ConceptOverride(BaseModel):
    concept_name: str
    original_type: str = ""
    human_type: str = ""
    original_domains: list[str] = Field(default_factory=list)
    human_domains: list[str] = Field(default_factory=list)
    additional_context: str = ""

class CycleReport(BaseModel):
    cycle_number: int
    timestamp: str
    papers_before: int = 0
    papers_after: int = 0
    concepts_before: int = 0
    concepts_after: int = 0
    bridges_before: int = 0
    bridges_after: int = 0
    hypotheses_generated: int = 0
    papers_auto_expanded: int = 0
    heal_actions: int = 0
    feedback_applied: bool = False
