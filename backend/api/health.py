"""Health / readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    engine = request.app.state.rag_engine
    documents = request.app.state.document_service
    llm = request.app.state.llm
    registry = getattr(request.app.state, "provider_registry", None)
    chain = llm.describe()
    provider_status = registry.describe()["providers"] if registry is not None else []
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm": {
            "providers": llm.provider_names,
            "skipped": llm.skipped_names,
            "primary": chain[0]["name"] if chain else None,
            "primary_model": chain[0]["model"] if chain else None,
            "details": chain,
            "status": provider_status,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": engine.stats().get("embedding_model"),
        },
        "chroma": {
            "mode": settings.chroma_mode,
            "collection": engine.stats().get("collection"),
        },
        "rag": engine.stats(),
        "documents": documents.stats(),
    }
