"""Pydantic v2 models for raw documents, entities, citations, claims, and methodology."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: list[str]
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str = ""
    full_text: str
    source_path: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    year: int | None = None
    domain_tags: list[str] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str
    entity_type: Literal["chemical", "gene", "method", "material", "phenomenon", "organism"]
    source_doc_id: str
    context_sentence: str


class Claim(BaseModel):
    statement: str
    supporting_evidence: list[str]
    source_doc_id: str
    section: str
    confidence: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    title: str
    authors: list[str]
    year: int | None = None
    doi: str | None = None
    bibtex: str = ""
