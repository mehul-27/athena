"""Document service: upload → validate → store → extract → chunk → embed → index.

Owns the on-disk document registry and the physical PDFs. Each document is keyed
by a content hash, which gives two things the spec asks for:

* **duplicate handling** — re-uploading the same bytes is detected and short-circuited
  (no duplicate vectors);
* **stable vectors** — chunk ids derive from `document_id` + text, so re-indexing
  is idempotent.

Deletion removes the physical file, the registry entry *and* the vectors, so no
orphan chunks are left behind.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import Settings
from rag.engine import RagEngine
from rag.ingestion import ingest_pdf

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "document.pdf")
    cleaned = _SAFE_NAME.sub("_", base).strip() or "document.pdf"
    return cleaned


class DocumentService:
    def __init__(self, settings: Settings, engine: RagEngine) -> None:
        self.settings = settings
        self.engine = engine
        self.docs_dir = settings.resolved_documents_dir()
        self.registry_file = settings.registry_path()
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Dict[str, Any]] = self._load_registry()

    # ------------------------------------------------------------------
    # Registry persistence
    # ------------------------------------------------------------------
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if not self.registry_file.exists():
            return {}
        try:
            data = json.loads(self.registry_file.read_text(encoding="utf-8"))
            docs = data.get("documents") if isinstance(data, dict) else None
            return docs if isinstance(docs, dict) else {}
        except Exception as exc:
            logger.warning("Could not read document registry (%s); starting empty", exc)
            return {}

    def _save_registry(self) -> None:
        payload = {"documents": self._registry}
        fd, tmp = tempfile.mkstemp(dir=str(self.registry_file.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.registry_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------------
    def list_documents(self) -> List[Dict[str, Any]]:
        return sorted(
            self._registry.values(),
            key=lambda d: d.get("created_at", ""),
        )

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self._registry.get(document_id)

    # ------------------------------------------------------------------
    def ingest_upload(
        self, filename: str, data: bytes, *, owner: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate, store and index an uploaded PDF."""
        display_name = _safe_filename(filename)
        logger.info("Upload received filename=%s bytes=%d", display_name, len(data))

        if not data:
            return {"success": False, "error": "Empty file"}
        if len(data) > self.settings.upload_max_bytes:
            return {
                "success": False,
                "error": f"File exceeds {self.settings.upload_max_bytes} byte limit",
            }
        if not data.startswith(b"%PDF"):
            logger.warning("Rejected upload filename=%s reason=not-a-pdf", display_name)
            return {"success": False, "error": "Only PDF files are supported"}

        content_hash = hashlib.sha256(data).hexdigest()
        document_id = content_hash[:16]

        existing = self._registry.get(document_id)
        if existing:
            logger.info(
                "Duplicate upload ignored filename=%s document_id=%s", display_name, document_id
            )
            return {
                "success": True,
                "duplicate": True,
                "document": existing,
                "message": "Document already indexed",
            }

        stored_name = f"{document_id}_{display_name}"
        stored_path = self.docs_dir / stored_name
        stored_path.write_bytes(data)
        logger.info("Stored document document_id=%s path=%s", document_id, stored_path)

        result = ingest_pdf(
            self.engine,
            document_id,
            str(stored_path),
            display_name,
            owner=owner,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )
        if not result.get("success"):
            stored_path.unlink(missing_ok=True)
            logger.warning("Ingestion failed document_id=%s: %s", document_id, result.get("message"))
            return {"success": False, "error": result.get("message", "Ingestion failed")}

        record = {
            "document_id": document_id,
            "filename": display_name,
            "stored_filename": stored_name,
            "stored_path": str(stored_path),
            "source": display_name,
            "content_hash": content_hash,
            "size_bytes": len(data),
            "chunks": result["chunks"],
            "chars": result["chars"],
            "owner": owner,
            "created_at": _now_iso(),
        }
        self._registry[document_id] = record
        self._save_registry()
        logger.info(
            "Indexing complete document_id=%s filename=%s chunks=%d total_docs=%d total_chunks=%d",
            document_id, display_name, result["chunks"], len(self._registry), self.engine.count(),
        )
        return {"success": True, "duplicate": False, "document": record}

    # ------------------------------------------------------------------
    def delete(self, document_id: str) -> Dict[str, Any]:
        record = self._registry.get(document_id)
        if not record:
            return {"success": False, "error": "Document not found", "removed_chunks": 0}

        removed = self.engine.delete_document(document_id)
        path = record.get("stored_path")
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError as exc:
                logger.warning("Could not delete file %s: %s", path, exc)
        self._registry.pop(document_id, None)
        self._save_registry()
        logger.info(
            "Deleted document document_id=%s filename=%s removed_chunks=%d",
            document_id, record.get("filename"), removed,
        )
        return {"success": True, "removed_chunks": removed, "document_id": document_id}

    def stats(self) -> Dict[str, Any]:
        return {
            "documents": len(self._registry),
            "chunks": self.engine.count(),
        }
