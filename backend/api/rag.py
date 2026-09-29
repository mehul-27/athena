"""RAG Chat API + document management (upload / list / delete)."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from backend.chat import rag_chat

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RagChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


@router.post("/chat")
def rag_chat_endpoint(req: RagChatRequest, request: Request) -> dict:
    settings = request.app.state.settings
    selection = request.app.state.provider_registry.mode_model("rag")
    return rag_chat.answer(
        req.message,
        request.app.state.rag_engine,
        request.app.state.llm,
        k=settings.rag_top_k,
        threshold=settings.rag_relevance_threshold,
        prefer=selection["provider"] or None,
        model=selection["model"] or None,
    )


@router.get("/documents")
def list_documents(request: Request) -> dict:
    service = request.app.state.document_service
    return {"documents": service.list_documents(), "stats": service.stats()}


@router.post("/documents")
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict:
    service = request.app.state.document_service
    data = await file.read()
    result = service.ingest_upload(file.filename or "document.pdf", data)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
    return result


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, request: Request) -> dict:
    service = request.app.state.document_service
    result = service.delete(document_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return result


@router.get("/stats")
def rag_stats(request: Request) -> dict:
    engine = request.app.state.rag_engine
    service = request.app.state.document_service
    return {"engine": engine.stats(), "documents": service.stats()}
