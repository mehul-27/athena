"""Athena application entry point.

Run with one of:
    python -m backend.main
    uvicorn backend.main:create_app --factory --host 127.0.0.1 --port 8000

The app is built by a factory (not at import time) so tests can construct an
isolated instance with their own settings/paths and a stub LLM.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import calendar, chat, conversations, health, prefs, providers, rag, research, search
from backend.config import ATHENA_ROOT, Settings, get_settings
from backend.conversations.store import ConversationStore
from backend.logging_config import configure_logging
from backend.providers.registry import ProviderRegistry
from backend.providers.router import build_llm_router
from backend.research.runner import ResearchRunner
from backend.services.documents import DocumentService
from rag.ingestion import build_engine

logger = logging.getLogger(__name__)

FRONTEND_DIR = ATHENA_ROOT / "frontend"


def create_app(
    settings: Optional[Settings] = None,
    *,
    llm=None,
    engine=None,
    document_service=None,
    registry: Optional[ProviderRegistry] = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    engine = engine or build_engine(settings)
    document_service = document_service or DocumentService(settings, engine)
    # The registry owns provider configuration + health and feeds the router
    # live, so Settings changes apply without a restart.
    registry = registry or ProviderRegistry(settings)
    llm = llm or build_llm_router(settings, registry=registry)
    conversation_store = ConversationStore(settings)

    app = FastAPI(title=settings.app_name, version="0.2.0")
    app.state.settings = settings
    app.state.rag_engine = engine
    app.state.document_service = document_service
    app.state.llm = llm
    app.state.provider_registry = registry
    app.state.conversation_store = conversation_store
    app.state.research_runner = ResearchRunner(
        settings, llm, registry=registry, conversations=conversation_store
    )

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api"):
            # Live configuration must never be served from a browser cache —
            # otherwise a provider/model change (or a restart) can look like it
            # "vanished" until the user hard-reloads.
            response.headers["Cache-Control"] = "no-store"
            logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
        return response

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(rag.router)
    app.include_router(search.router)
    app.include_router(research.router)
    app.include_router(providers.router)
    app.include_router(prefs.router)
    app.include_router(calendar.router)
    app.include_router(conversations.router)

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    logger.info(
        "Athena ready chroma=%s embeddings=%s collection=%s llm_providers=%s",
        settings.chroma_mode, settings.embedding_provider,
        settings.rag_collection, ",".join(llm.provider_names) or "none",
    )
    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(create_app(settings), host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()
