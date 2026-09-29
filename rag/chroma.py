"""ChromaDB wiring.

Athena defaults to an **embedded** `PersistentClient` so the app runs standalone
with no external service, unlike Odysseus's required standalone ChromaDB server
(`src/chroma_client.py`). An HTTP mode is still available via `CHROMA_MODE=http`.

We keep Odysseus's collection-metadata idea: the collection records which
embedding model/dimension built it, so a later config change is detectable
instead of silently producing dimension-mismatch errors.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from backend.config import Settings

logger = logging.getLogger(__name__)


def build_chroma_client(settings: Settings):
    import chromadb

    mode = (settings.chroma_mode or "embedded").lower()
    if mode == "http":
        logger.info("ChromaDB HTTP client host=%s port=%s", settings.chroma_host, settings.chroma_port)
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    path = str(settings.resolved_chroma_path())
    logger.info("ChromaDB embedded PersistentClient path=%s", path)
    return chromadb.PersistentClient(path=path)


def embedding_fingerprint(model: str, url: str, dimension: int) -> str:
    raw = f"{url}\n{model}\n{dimension}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_or_create_collection(client, name: str, *, model: str, url: str, dimension: int):
    metadata = {
        "hnsw:space": "cosine",
        "embedding_model": model,
        "embedding_url": url,
        "embedding_dimension": dimension,
        "embedding_fingerprint": embedding_fingerprint(model, url, dimension),
    }
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.get_or_create_collection(name=name, metadata=metadata)
        return collection

    existing = collection.metadata or {}
    fp = existing.get("embedding_fingerprint")
    if fp not in (None, metadata["embedding_fingerprint"]):
        logger.warning(
            "Collection %s was built with a different embedding model "
            "(fingerprint %s -> %s). Retrieval may be inconsistent until re-index.",
            name, fp, metadata["embedding_fingerprint"],
        )
    return collection
