"""Settings API for LLM providers.

The primary runtime configuration surface. Everything here is generic: a provider
can be created, edited, tested, reordered and removed without existing anywhere in
Athena's source code. Presets (`GET` payload → `presets`) only pre-fill defaults.

Secrets never cross this boundary: responses carry only `has_key`, a masked hint
(`key_masked`) and a non-secret fingerprint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])


class CreateProviderRequest(BaseModel):
    # `name` and `base_url` are the only things the user must supply; everything
    # else is a preset/default. There is no allowed-value list for `name`.
    name: str = Field(default="", max_length=120)
    base_url: str = Field(..., min_length=1, max_length=2000)
    preset: Optional[str] = ""
    api_key: Optional[str] = None
    model: Optional[str] = None
    auth_type: Optional[str] = None
    api_key_required: Optional[bool] = None
    provider_type: Optional[str] = None
    enabled: Optional[bool] = True


class UpdateProviderRequest(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    enabled: Optional[bool] = None
    auth_type: Optional[str] = None
    api_key_required: Optional[bool] = None


class OrderRequest(BaseModel):
    order: List[str] = Field(default_factory=list)


class TestProviderRequest(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None


class TestDraftRequest(BaseModel):
    preset: Optional[str] = ""
    name: Optional[str] = ""
    base_url: str = Field(..., min_length=1, max_length=2000)
    api_key: Optional[str] = None
    model: Optional[str] = None
    auth_type: Optional[str] = None
    api_key_required: Optional[bool] = None


class DiscoverRequest(BaseModel):
    api_key: Optional[str] = None


class ModeModelRequest(BaseModel):
    mode: str = Field(..., min_length=1)
    provider: Optional[str] = ""
    model: Optional[str] = ""


class ResearchModelsRequest(BaseModel):
    fast: Optional[str] = None
    strong: Optional[str] = None


def _registry(request: Request):
    return request.app.state.provider_registry


def _get_or_404(registry, provider_id: str) -> str:
    if not registry.has(provider_id):
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")
    return provider_id


@router.get("")
def list_providers(request: Request) -> dict:
    """Configured providers (with status) plus the available presets."""
    return _registry(request).describe()


@router.get("/presets")
def list_presets(request: Request) -> dict:
    return {"presets": _registry(request).presets()}


@router.post("")
def create_provider(body: CreateProviderRequest, request: Request) -> dict:
    """Add a provider — any OpenAI-compatible endpoint, preset or not."""
    try:
        row = _registry(request).create(
            name=body.name,
            preset=body.preset or "",
            base_url=body.base_url,
            api_key=body.api_key or "",
            model=body.model or "",
            auth_type=body.auth_type or "",
            api_key_required=body.api_key_required,
            provider_type=body.provider_type or "",
            enabled=True if body.enabled is None else body.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"provider": row}


@router.patch("/{provider_id}")
def update_provider(provider_id: str, body: UpdateProviderRequest, request: Request) -> dict:
    registry = _registry(request)
    _get_or_404(registry, provider_id)
    try:
        row = registry.update(
            provider_id,
            name=body.name,
            base_url=body.base_url,
            api_key=body.api_key,
            model=body.model,
            enabled=body.enabled,
            auth_type=body.auth_type,
            api_key_required=body.api_key_required,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"provider": row}


@router.delete("/{provider_id}")
def remove_provider(provider_id: str, request: Request) -> dict:
    registry = _registry(request)
    _get_or_404(registry, provider_id)
    removed = registry.remove(provider_id)
    return {"removed": removed, "id": provider_id}


@router.put("/order")
def set_order(body: OrderRequest, request: Request) -> dict:
    registry = _registry(request)
    try:
        order = registry.set_order(body.order)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {exc.args[0]}") from exc
    return {"order": order}


@router.put("/mode-models")
def set_mode_model(body: ModeModelRequest, request: Request) -> dict:
    try:
        modes = _registry(request).set_mode_model(
            body.mode, provider=body.provider or "", model=body.model or ""
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {exc.args[0]}") from exc
    return {"mode_models": modes}


@router.put("/research-models")
def set_research_models(body: ResearchModelsRequest, request: Request) -> dict:
    return {"research_models": _registry(request).set_research_models(fast=body.fast, strong=body.strong)}


@router.post("/test")
def test_draft(body: TestDraftRequest, request: Request) -> dict:
    """Test a provider that has not been saved yet (the Add form's Test button)."""
    return _registry(request).test_draft(
        preset=body.preset or "",
        name=body.name or "",
        base_url=body.base_url,
        api_key=body.api_key or "",
        model=body.model or "",
        auth_type=body.auth_type or "",
        api_key_required=body.api_key_required,
    )


@router.post("/models")
def discover_draft(body: TestDraftRequest, request: Request) -> dict:
    """Model discovery for a provider that has not been saved yet."""
    models = _registry(request).discover_models_draft(
        preset=body.preset or "",
        base_url=body.base_url,
        api_key=body.api_key or "",
        model=body.model or "",
        auth_type=body.auth_type or "",
        api_key_required=body.api_key_required,
    )
    return {"models": models, "count": len(models), "supported": bool(models)}


@router.post("/{provider_id}/test")
def test_provider(provider_id: str, request: Request, body: Optional[TestProviderRequest] = None) -> dict:
    registry = _registry(request)
    _get_or_404(registry, provider_id)
    payload = body or TestProviderRequest()
    result = registry.test(provider_id, api_key=payload.api_key, model=payload.model)
    return {"id": provider_id, **result, "provider": registry.row(provider_id)}


@router.get("/{provider_id}/models")
def cached_models(provider_id: str, request: Request) -> dict:
    registry = _registry(request)
    _get_or_404(registry, provider_id)
    return {"id": provider_id, "models": registry.row(provider_id).get("models", [])}


@router.post("/{provider_id}/models")
def discover_models(provider_id: str, request: Request, body: Optional[DiscoverRequest] = None) -> dict:
    registry = _registry(request)
    _get_or_404(registry, provider_id)
    payload = body or DiscoverRequest()
    models = registry.discover_models(provider_id, api_key=payload.api_key)
    return {"id": provider_id, "models": models, "count": len(models), "supported": bool(models)}
