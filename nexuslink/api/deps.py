"""FastAPI dependency providers — all routes import from here."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from nexuslink.config import NexusConfig
from nexuslink.main import NexusLink

# -------------------------------------------------------------------------
# Config — cached singleton; reads .env once per process
# -------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> NexusConfig:
    return NexusConfig()


# -------------------------------------------------------------------------
# NexusLink — lazy singleton; created on first request
# -------------------------------------------------------------------------

_nexuslink: NexusLink | None = None


def get_nexuslink(
    config: Annotated[NexusConfig, Depends(get_config)],
) -> NexusLink:
    global _nexuslink
    if _nexuslink is None:
        _nexuslink = NexusLink(vault_path=config.vault_path, config=config)
    return _nexuslink
