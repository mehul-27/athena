"""Embedding providers.

Adapted from Odysseus `src/embeddings.py` (`EmbeddingClient` HTTP + `FastEmbedClient`
local ONNX), reduced to a single-lane factory. Odysseus's dual-lane scheme
(`src/embedding_lanes.py`) exists to keep a user-configured HTTP model separate
from the local FastEmbed fallback so two different vector dimensions never share
one Chroma collection. Athena picks exactly one provider from `.env`, so that
machinery is unnecessary — but we keep the collection-dimension fingerprint as a
safety check.

The Windows guards below are carried over verbatim in spirit: the repo lives on
`M:\\` and HuggingFace's default symlinked cache fails there
(`[WinError 1463] symbolic link cannot be followed`). They must be set before
`huggingface_hub` is first imported, hence module-top.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List, Optional, Protocol, Sequence

import numpy as np

from backend.config import Settings
from rag.text import tokenize

if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logger = logging.getLogger(__name__)

DEFAULT_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HTTP_MODEL = "all-minilm:l6-v2"
HASHING_DIM = 256


class EmbeddingProvider(Protocol):
    model: str
    url: str

    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(vecs: np.ndarray) -> np.ndarray:
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


class FastEmbedProvider:
    """Local ONNX embeddings via `fastembed`. No service, works offline."""

    def __init__(self, model: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        from fastembed import TextEmbedding

        self.model = model or DEFAULT_FASTEMBED_MODEL
        self.url = "local://fastembed"
        cache = cache_dir or os.path.join(os.getcwd(), "data", "fastembed_cache")
        os.makedirs(cache, exist_ok=True)
        self._embedding = TextEmbedding(model_name=self.model, cache_dir=cache)
        self._dim: Optional[int] = None
        logger.info("FastEmbed loaded model=%s cache=%s", self.model, cache)

    def dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["hello"]).shape[1])
            logger.info("Embedding dimension=%d (model=%s)", self._dim, self.model)
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension()), dtype="float32")
        vecs = np.array(list(self._embedding.embed(list(texts))), dtype="float32")
        return _normalize(vecs)


class HttpEmbeddingProvider:
    """OpenAI-compatible `/v1/embeddings` over HTTP."""

    def __init__(
        self,
        url: str,
        model: Optional[str] = None,
        api_key: str = "",
        *,
        batch_size: int = 16,
        max_chars: int = 900,
    ) -> None:
        import httpx

        self.url = url
        self.model = model or DEFAULT_HTTP_MODEL
        self.api_key = api_key
        self._batch_size = max(1, batch_size)
        self._max_chars = max(200, max_chars)
        self._dim: Optional[int] = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=3.0)
        )

    def _post(self, batch: List[str]) -> List[List[float]]:
        resp = self._client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={"input": batch, "model": self.model},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("data", [])
        embeddings.sort(key=lambda e: e.get("index", 0))
        return [e["embedding"] for e in embeddings]

    def dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["hello"]).shape[1])
            logger.info("Embedding dimension=%d (model=%s url=%s)", self._dim, self.model, self.url)
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension()), dtype="float32")
        texts = [t[: self._max_chars] for t in texts]
        out: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(self._post(texts[i : i + self._batch_size]))
        return _normalize(np.array(out, dtype="float32"))


class HashingEmbeddingProvider:
    """Deterministic, offline bag-of-words embedding (feature hashing).

    Used by the default test suite and available as `EMBEDDING_PROVIDER=hashing`
    for a zero-download run. It drops stop words, so two texts that share no
    content words are orthogonal — which makes the relevance threshold's
    positive/negative separation deterministic and reproducible.
    """

    def __init__(self, dim: int = HASHING_DIM) -> None:
        self.model = f"hashing-{dim}"
        self.url = "local://hashing"
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        out = np.zeros((len(texts), self._dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                idx = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16) % self._dim
                out[i, idx] += 1.0
        return _normalize(out)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = (settings.embedding_provider or "fastembed").lower()
    if provider == "http":
        if not settings.embedding_url:
            raise RuntimeError("EMBEDDING_PROVIDER=http requires EMBEDDING_URL")
        return HttpEmbeddingProvider(
            settings.embedding_url,
            model=settings.embedding_model or None,
            api_key=settings.embedding_api_key,
            batch_size=settings.embedding_batch_size,
            max_chars=settings.embedding_max_chars,
        )
    if provider == "hashing":
        logger.info("Using deterministic hashing embeddings (offline mode)")
        return HashingEmbeddingProvider()
    if provider == "fastembed":
        return FastEmbedProvider(
            model=settings.fastembed_model or None,
            cache_dir=str(settings.resolved_fastembed_cache()),
        )
    raise RuntimeError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
