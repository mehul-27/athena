"""Deterministic retrieval tests — the mandatory Project Lumen check."""

from rag.ingestion import build_engine, ingest_pdf
from rag.retrieval import hybrid_search, retrieve
from tests.conftest import LUMEN_PDF

QUERY = "What is the primary validation code for Project Lumen?"


def _engine_with_lumen(settings):
    engine = build_engine(settings)
    ingest_pdf(engine, "lumen-test", str(LUMEN_PDF), "Project Lumen.pdf")
    return engine


def test_project_lumen_retrieval_returns_chunk_with_code(settings):
    engine = _engine_with_lumen(settings)
    results = hybrid_search(engine, QUERY, k=5)
    assert results, "expected at least one retrieved chunk"
    joined = " ".join(r["document"] for r in results)
    assert "LUMEN-5831-ORBIT" in joined, "retrieved chunks must contain the validation code"


def test_retrieval_with_threshold_keeps_relevant_chunk(settings):
    engine = _engine_with_lumen(settings)
    kept, diag = retrieve(engine, QUERY, k=5, threshold=0.25)
    assert kept, "relevant query must clear the threshold"
    assert diag["best_score"] >= 0.25
    assert any("LUMEN-5831-ORBIT" in r["document"] for r in kept)


def test_retrieval_sources_report_filename_and_chunk(settings):
    engine = _engine_with_lumen(settings)
    kept, _ = retrieve(engine, QUERY, k=5, threshold=0.25)
    meta = kept[0]["metadata"]
    assert meta["filename"] == "Project Lumen.pdf"
    assert "chunk_id" in meta


def test_offtopic_query_scores_below_threshold(settings):
    engine = _engine_with_lumen(settings)
    kept, diag = retrieve(engine, "What is the capital of France?", k=5, threshold=0.25)
    assert kept == [], f"off-topic query should be rejected (best_score={diag['best_score']})"
