"""RagEngine — the vector store layer.

Owns a Chroma collection plus its embedding provider. Adapted from Odysseus
`src/rag_vector.py::VectorRAG` (content-addressed ids, redundant-add skip,
delete-by-metadata) but trimmed of that app's multi-lane and multi-tenant
machinery.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence

from rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

ID_PREFIX = "doc_"


def chunk_id(document_id: str, text: str) -> str:
    """Content-addressed chunk id.

    Stable across re-indexing, so identical chunks never double-index. Scoped by
    `document_id` so two documents that share a chunk keep separate rows.
    (Same idea as Odysseus's `_generate_doc_id`.)
    """
    digest = hashlib.sha256(f"{document_id}\x00{text}".encode("utf-8")).hexdigest()[:16]
    return f"{ID_PREFIX}{digest}"


class RagEngine:
    def __init__(self, collection, embeddings: EmbeddingProvider) -> None:
        self._collection = collection
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    @property
    def collection(self):
        return self._collection

    @property
    def dimension(self) -> int:
        return self._embeddings.dimension()

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        return {
            "collection": self._collection.name,
            "count": self.count(),
            "embedding_model": self._embeddings.model,
            "embedding_url": self._embeddings.url,
            "dimension": self._embeddings.dimension(),
        }

    # ------------------------------------------------------------------
    def add_chunks(
        self,
        document_id: str,
        chunks: Sequence[str],
        *,
        filename: str,
        source: str,
        owner: Optional[str] = None,
    ) -> Dict[str, int]:
        """Embed + index chunks. Skips ids that already exist (idempotent)."""
        if not chunks:
            return {"added": 0, "skipped": 0}

        ids = [chunk_id(document_id, c) for c in chunks]
        metadatas = []
        for i, c in enumerate(chunks):
            meta: Dict[str, Any] = {
                "document_id": document_id,
                "filename": filename,
                "source": source,
                "chunk_id": i,
                "chunk_chars": len(c),
            }
            if owner:
                meta["owner"] = owner
            metadatas.append(meta)

        try:
            existing = set(self._collection.get(ids=ids).get("ids") or [])
        except Exception:
            existing = set()

        new_idx = [i for i, cid in enumerate(ids) if cid not in existing]
        if not new_idx:
            return {"added": 0, "skipped": len(ids)}

        new_texts = [chunks[i] for i in new_idx]
        embeddings = self._embeddings.encode(new_texts).tolist()
        self._collection.add(
            ids=[ids[i] for i in new_idx],
            embeddings=embeddings,
            documents=new_texts,
            metadatas=[metadatas[i] for i in new_idx],
        )
        return {"added": len(new_idx), "skipped": len(ids) - len(new_idx)}

    def query(self, query_text: str, k: int) -> Dict[str, Any]:
        """Raw vector query. Returns Chroma's nested result shape."""
        total = self.count()
        if total == 0 or not query_text.strip():
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        n = max(1, min(k, total))
        vec = self._embeddings.encode([query_text]).tolist()
        return self._collection.query(
            query_embeddings=vec,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk for a document. Returns the count removed."""
        try:
            got = self._collection.get(where={"document_id": document_id}, include=[])
        except Exception as exc:
            logger.warning("delete lookup failed for document_id=%s: %s", document_id, exc)
            return 0
        ids = got.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        """Drop and recreate the collection (used by tests)."""
        name = self._collection.name
        client = getattr(self._collection, "_client", None)
        if client is None:
            return
        try:
            client.delete_collection(name)
        except Exception:
            pass
        self._collection = client.get_or_create_collection(name=name)

    def list_chunks(self) -> List[Dict[str, Any]]:
        data = self._collection.get(include=["documents", "metadatas"])
        rows = []
        for i, cid in enumerate(data.get("ids") or []):
            rows.append(
                {
                    "id": cid,
                    "document": (data.get("documents") or [])[i],
                    "metadata": (data.get("metadatas") or [])[i],
                }
            )
        return rows
