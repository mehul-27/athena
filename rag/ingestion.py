"""PDF extraction + chunking + indexing.

Extraction uses `pypdf` (adapted from Odysseus `src/personal_docs.py::extract_pdf_text`)
— simple, permissive-licensed, already proven. Chunking is the sentence-boundary
aware splitter from Odysseus `src/rag_vector.py::_split_into_chunks`, with the
long-sentence hard-split and overlap-tail behaviour preserved.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence

from rag.engine import RagEngine

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def extract_pdf_text(path: str) -> str:
    """Extract text from a PDF using pypdf. Returns '' on failure."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("PDF extraction failed for %s: %s", path, exc)
        return ""


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    """Sentence-aware chunking with sentence overlap."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    step = max(1, size - overlap)

    for sentence in sentences:
        if len(sentence) > size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            for start in range(0, len(sentence), step):
                chunks.append(sentence[start : start + size])
            continue

        if current_len + len(sentence) + 1 > size and current:
            chunks.append(" ".join(current))
            tail: List[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) + 1 > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, current_len = tail, tail_len

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def ingest_pdf(
    engine: RagEngine,
    document_id: str,
    path: str,
    filename: str,
    *,
    owner: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """Extract → chunk → embed → index a PDF. Logs each stage."""
    logger.info("Extraction start document_id=%s file=%s", document_id, filename)
    text = extract_pdf_text(path)
    logger.info("Extraction complete document_id=%s chars=%d", document_id, len(text))
    if not text.strip():
        return {"success": False, "chunks": 0, "added": 0, "chars": 0, "message": "No extractable text"}

    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
    logger.info("Chunking complete document_id=%s chunks=%d", document_id, len(chunks))

    dim = engine.dimension
    added = engine.add_chunks(
        document_id,
        chunks,
        filename=filename,
        source=filename,
        owner=owner,
    )
    logger.info(
        "Embedding+indexing complete document_id=%s chunks=%d added=%d skipped=%d dim=%d",
        document_id, len(chunks), added["added"], added["skipped"], dim,
    )
    return {
        "success": True,
        "chars": len(text),
        "chunks": len(chunks),
        "added": added["added"],
        "skipped": added["skipped"],
        "dimension": dim,
    }


def build_engine(settings) -> RagEngine:
    """Construct the RagEngine from settings (client + embeddings + collection)."""
    from rag.chroma import build_chroma_client, get_or_create_collection
    from rag.embeddings import build_embedding_provider

    embeddings = build_embedding_provider(settings)
    dimension = embeddings.dimension()
    client = build_chroma_client(settings)
    collection = get_or_create_collection(
        client,
        settings.rag_collection,
        model=embeddings.model,
        url=embeddings.url,
        dimension=dimension,
    )
    logger.info(
        "RAG engine ready collection=%s dimension=%d docs=%d",
        collection.name, dimension, collection.count(),
    )
    return RagEngine(collection, embeddings)
