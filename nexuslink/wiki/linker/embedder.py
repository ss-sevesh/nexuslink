"""Concept embedder: maps ExtractedEntity objects to dense vectors.

Uses Ollama (nomic-embed-text) when available for better cross-domain semantic
understanding; falls back to sentence-transformers (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import hashlib
import json
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from nexuslink.raw.schemas.models import ExtractedEntity

_WIKI_DIR = Path(__file__).parent.parent          # wiki/
_CACHE_DIR = _WIKI_DIR / ".cache"
_EMB_PATH = _CACHE_DIR / "embeddings.npz"
_IDX_PATH = _CACHE_DIR / "embeddings_index.json"

_DEFAULT_MODEL = "all-mpnet-base-v2"
_OLLAMA_EMBED_MODEL = "nomic-embed-text"


def _ollama_available() -> bool:
    """Return True if Ollama is reachable and nomic-embed-text is installed."""
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
            return any(_OLLAMA_EMBED_MODEL in m["name"] for m in data.get("models", []))
    except Exception:
        return False


def _ollama_embed(texts: list[str]) -> list[np.ndarray]:
    """Call Ollama embedding API; returns list of normalised float32 arrays."""
    import urllib.request, urllib.error
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host}/api/embed"
    results = []
    for text in texts:
        body = json.dumps({"model": _OLLAMA_EMBED_MODEL, "input": text}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        vec = np.array(data["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        results.append(vec / norm if norm > 0 else vec)
    return results


class ConceptEmbedder:
    """Embeds concept names + context sentences and caches results to disk."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._use_ollama: bool = False  # nomic-embed-text collapses short concept names
        if self._use_ollama:
            logger.info("ConceptEmbedder using Ollama/{} for embeddings", _OLLAMA_EMBED_MODEL)
        else:
            logger.info("ConceptEmbedder using sentence-transformers/{}", model_name)
        # hash(text) -> np.ndarray
        self._cache: dict[str, np.ndarray] = {}
        # hash -> original text (for persistence round-trips)
        self._key_index: dict[str, str] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_entity(self, entity: ExtractedEntity) -> np.ndarray:
        """Return a unit-normalised embedding for *entity*."""
        text = _entity_text(entity)
        h = _sha(text)
        if h not in self._cache:
            logger.debug("Embedding entity {!r}", entity.name)
            if getattr(self, "_use_ollama", False):
                emb = _ollama_embed([text])[0]
            else:
                emb = self._get_model().encode(text, normalize_embeddings=True)
            self._cache[h] = emb
            self._key_index[h] = text
        return self._cache[h]

    def embed_batch(self, entities: list[ExtractedEntity]) -> dict[str, np.ndarray]:
        """Embed a list of entities, returning ``{entity.name: embedding}``.

        Cache hits are served without a model call; remaining texts are sent to
        the model in a single batched encode call.
        """
        result: dict[str, np.ndarray] = {}
        pending_texts: list[str] = []
        pending_meta: list[tuple[str, str, str]] = []  # (name, hash, text)

        for entity in entities:
            text = _entity_text(entity)
            h = _sha(text)
            if h in self._cache:
                result[entity.name] = self._cache[h]
            else:
                pending_texts.append(text)
                pending_meta.append((entity.name, h, text))

        if pending_texts:
            logger.debug("Batch-embedding {} new entities", len(pending_texts))
            if getattr(self, "_use_ollama", False):
                embeddings = _ollama_embed(pending_texts)
            else:
                embeddings = list(self._get_model().encode(
                    pending_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
                ))
            for (name, h, text), emb in zip(pending_meta, embeddings):
                self._cache[h] = emb
                self._key_index[h] = text
                result[name] = emb

        return result

    def save_cache(self) -> None:
        """Persist the embedding cache to *wiki/.cache/embeddings.npz*."""
        if not self._cache:
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_EMB_PATH, **self._cache)
        _IDX_PATH.write_text(json.dumps(self._key_index), encoding="utf-8")
        logger.info("Saved {} embeddings to cache", len(self._cache))

    async def save_cache_async(self) -> None:
        """Async wrapper around :meth:`save_cache`."""
        await asyncio.to_thread(self.save_cache)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model: {}", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _load_cache(self) -> None:
        if not (_EMB_PATH.exists() and _IDX_PATH.exists()):
            return
        try:
            data = np.load(_EMB_PATH)
            key_index: dict[str, str] = json.loads(_IDX_PATH.read_text(encoding="utf-8"))
            self._cache = {h: data[h] for h in data.files}
            self._key_index = key_index
            logger.debug("Loaded {} embeddings from cache", len(self._cache))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load embedding cache: {}", exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _entity_text(entity: ExtractedEntity) -> str:
    return f"{entity.name}: {entity.context_sentence}"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
