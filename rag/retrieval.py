"""Hybrid retrieval + relevance gating + grounded-context assembly.

The scoring formula (0.7 * cosine similarity + 0.3 * keyword overlap) is lifted
from Odysseus `src/rag_vector.py::VectorRAG.search`. Athena adds an explicit,
mandatory **relevance threshold**: RAG Chat only answers when at least one chunk
clears it. That gate is what makes the "not found in documents" behaviour honest
rather than a prompt suggestion the model can ignore.

Note: the score is a *relative* hybrid score, not a calibrated probability. The
threshold is a tunable gate (see README), validated by the retrieval tests.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

from rag.engine import RagEngine
from rag.text import token_set

logger = logging.getLogger(__name__)

VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


def normalize_query(query: str) -> str:
    return " ".join((query or "").split())


def hybrid_search(engine: RagEngine, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Vector search + keyword-overlap re-ranking over the same candidates."""
    over_fetch = max(k * 3, k + 5)
    raw = engine.query(query, over_fetch)

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    q_tokens = token_set(query)
    candidates: List[Dict[str, Any]] = []
    for i, doc_id in enumerate(ids):
        doc_text = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        distance = dists[i] if i < len(dists) else 1.0

        vector_sim = 1.0 - float(distance)
        if math.isnan(vector_sim):
            vector_sim = 0.0
        vector_sim = max(0.0, min(1.0, vector_sim))

        doc_tokens = token_set(doc_text)
        overlap = len(q_tokens & doc_tokens)
        keyword_score = overlap / len(q_tokens) if q_tokens else 0.0

        similarity = VECTOR_WEIGHT * vector_sim + KEYWORD_WEIGHT * keyword_score
        candidates.append(
            {
                "id": doc_id,
                "document": doc_text,
                "metadata": meta or {},
                "distance": round(float(distance), 4),
                "vector_similarity": round(vector_sim, 4),
                "keyword_score": round(keyword_score, 4),
                "similarity": round(similarity, 4),
            }
        )

    candidates.sort(key=lambda c: c["similarity"], reverse=True)

    # De-duplicate by id, preserving order, then cap at k.
    seen = set()
    out = []
    for row in candidates:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        out.append(row)
        if len(out) >= k:
            break
    return out


def retrieve(
    engine: RagEngine, query: str, k: int = 5, threshold: float = 0.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (relevant_chunks, diagnostics). Chunks below threshold are dropped."""
    query = normalize_query(query)
    candidates = hybrid_search(engine, query, k)
    kept = [c for c in candidates if c["similarity"] >= threshold]
    diag = {
        "candidates": len(candidates),
        "kept": len(kept),
        "threshold": threshold,
        "best_score": candidates[0]["similarity"] if candidates else 0.0,
    }
    return kept, diag


def build_context(results: List[Dict[str, Any]]) -> str:
    """Assemble the grounded context block passed to the LLM."""
    blocks = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata") or {}
        name = meta.get("filename") or meta.get("source") or "document"
        chunk_no = meta.get("chunk_id", "?")
        blocks.append(f"[{i}] {name} (chunk {chunk_no})\n{r['document']}")
    return "\n\n".join(blocks)


def sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Source list for the answer's Sources section."""
    out = []
    for r in results:
        meta = r.get("metadata") or {}
        out.append(
            {
                "filename": meta.get("filename") or meta.get("source") or "document",
                "document_id": meta.get("document_id"),
                "chunk_id": meta.get("chunk_id"),
                "score": r.get("similarity"),
            }
        )
    return out
