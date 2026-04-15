"""NexusLink runtime configuration — loads from .env and environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NexusConfig(BaseSettings):
    """All tuneable parameters for the NexusLink pipeline.

    Variables can be set in a ``.env`` file or as environment variables.
    Prefixed form (NEXUSLINK_*) takes priority; unprefixed fallbacks are also checked.

    Example .env::

        ANTHROPIC_API_KEY=sk-ant-...
        OLLAMA_MODEL=llama3.1:8b
        NEXUSLINK_SIMILARITY_THRESHOLD=0.70
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_prefix="NEXUSLINK_",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Paths
    vault_path: Path = Field(
        default=Path(__file__).parent / "wiki",
        description="Absolute or relative path to the Obsidian vault (wiki/) directory.",
    )

    # Anthropic — accepts NEXUSLINK_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NEXUSLINK_ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY",
        ),
    )

    # Ollama — accepts NEXUSLINK_OLLAMA_MODEL or OLLAMA_MODEL
    ollama_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NEXUSLINK_OLLAMA_MODEL",
            "OLLAMA_MODEL",
        ),
    )

    # Embeddings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model for concept embedding.",
    )

    # Bridge finding
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to form a cross-domain bridge.",
    )

    # Hypothesis generation
    top_n_hypotheses: int = Field(
        default=5,
        ge=1,
        description="Number of top hypotheses to refine and include in the report.",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Loguru log level: DEBUG, INFO, WARNING, ERROR.",
    )
