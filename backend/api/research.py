"""Deep Research API: async jobs, SSE progress, library and visual HTML report."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/research", tags=["research"])
logger = logging.getLogger(__name__)


class ResearchStartRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    max_rounds: Optional[int] = Field(default=None, ge=1, le=20)
    max_time: Optional[int] = Field(default=None, ge=1, le=86400)
    search_provider: Optional[str] = None
    category: Optional[str] = None
    # The chat this research belongs to (Deep Research is launched from Chat).
    conversation_id: Optional[str] = Field(default=None, max_length=200)


class ArchiveRequest(BaseModel):
    archived: bool = True


class HideImageRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=4096)


def _runner(request: Request):
    return request.app.state.research_runner


@router.post("/start")
async def start_research(req: ResearchStartRequest, request: Request) -> dict:
    runner = _runner(request)
    conversation_id = req.conversation_id
    if conversation_id:
        # Make sure the conversation exists so the result card has somewhere to go.
        conversation_id = request.app.state.conversation_store.ensure(conversation_id)["id"]
    return runner.start(
        req.query.strip(),
        max_rounds=req.max_rounds,
        max_time=req.max_time,
        search_provider=req.search_provider,
        category=req.category,
        conversation_id=conversation_id,
    )


@router.get("/active")
def active_research(request: Request) -> dict:
    runner = _runner(request)
    active = [
        {"research_id": research_id, **runner.status(research_id)}
        for research_id, entry in runner.active.items()
        if entry.get("status") == "running" and runner.status(research_id)
    ]
    return {"active": active}


@router.get("/status/{research_id}")
def research_status(research_id: str, request: Request) -> dict:
    status = _runner(request).status(research_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Research not found")
    return status


@router.post("/cancel/{research_id}")
def cancel_research(research_id: str, request: Request) -> dict:
    return {"cancelled": _runner(request).cancel(research_id)}


@router.get("/stream/{research_id}")
async def research_stream(research_id: str, request: Request):
    runner = _runner(request)
    if runner.status(research_id) is None:
        raise HTTPException(status_code=404, detail="Research not found")

    async def generate():
        last_progress = None
        while True:
            if await request.is_disconnected():
                return
            status = runner.status(research_id)
            if status is None:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                return
            state = status.get("status", "")
            progress = status.get("progress", {})
            if progress != last_progress:
                last_progress = progress
                yield f"data: {json.dumps({**progress, 'status': state})}\n\n"
            if state != "running":
                final = {"status": state, "final": True}
                result = runner.result(research_id)
                if state == "error" and result and result.get("result"):
                    final["error"] = str(result["result"])[:500]
                yield f"data: {json.dumps(final)}\n\n"
                return
            await asyncio.sleep(1.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/result/{research_id}")
def research_result(research_id: str, request: Request) -> dict:
    result = _runner(request).result(research_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research result not found")
    return result


@router.post("/result-peek/{research_id}")
def research_result_peek(research_id: str, request: Request) -> dict:
    return research_result(research_id, request)


@router.get("/report/{research_id}")
def research_report(research_id: str, request: Request) -> HTMLResponse:
    html = _runner(request).report_html(research_id)
    if html is None:
        raise HTTPException(status_code=404, detail="No visual report available")
    return HTMLResponse(content=html)


@router.get("/library")
def research_library(request: Request, archived: Optional[bool] = None, limit: int = 50) -> dict:
    rows = _runner(request).list(archived=archived, limit=min(max(limit, 1), 200))
    return {"research": rows, "total": len(rows)}


@router.get("/detail/{research_id}")
def research_detail(research_id: str, request: Request) -> dict:
    record = _runner(request).get(research_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Research not found")
    return record


@router.post("/{research_id}/archive")
def archive_research(research_id: str, body: ArchiveRequest, request: Request) -> dict:
    if not _runner(request).update(research_id, archived=body.archived):
        raise HTTPException(status_code=404, detail="Research not found")
    return {"archived": body.archived}


@router.delete("/{research_id}")
def delete_research(research_id: str, request: Request) -> dict:
    deleted = _runner(request).delete(research_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Research not found")
    return {"deleted": True}


@router.post("/{research_id}/hide-image")
def hide_image(research_id: str, body: HideImageRequest, request: Request) -> dict:
    record = _runner(request).get(research_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Research not found")
    hidden = list(record.get("hidden_images") or [])
    if body.url not in hidden:
        hidden.append(body.url)
    _runner(request).update(research_id, hidden_images=hidden)
    return {"ok": True}


@router.post("/{research_id}/unhide-images")
def unhide_images(research_id: str, request: Request) -> dict:
    if not _runner(request).update(research_id, hidden_images=[]):
        raise HTTPException(status_code=404, detail="Research not found")
    return {"ok": True}
