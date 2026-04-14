"""NexusLink FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from nexuslink.api.deps import get_config, get_nexuslink
from nexuslink.api.routes import graph, hypothesis, ingest


# ---------------------------------------------------------------------------
# Lifespan: warm up the NexusLink singleton at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    logger.info("Starting NexusLink API (vault={})", config.vault_path)
    # Force singleton creation so import errors surface at boot, not on first request
    get_nexuslink(config)
    yield
    logger.info("NexusLink API shutting down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NexusLink API",
    description="Cross-domain research hypothesis engine — RAW → WIKI → LLM pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(ingest.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(hypothesis.router, prefix="/api")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, Any]:
    """Returns 200 when the service is up."""
    config = get_config()
    return {
        "status": "ok",
        "vault_path": str(config.vault_path),
        "embedding_model": config.embedding_model,
    }


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("FileNotFoundError: {}", exc)
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError: {}", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    logger.error("RuntimeError: {}", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
