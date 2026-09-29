"""Extraction, chunking and indexing (unit-level, no HTTP)."""

from rag.engine import chunk_id
from rag.ingestion import build_engine, chunk_text, extract_pdf_text, ingest_pdf
from tests.conftest import LUMEN_PDF


def test_pdf_extraction_contains_validation_code():
    text = extract_pdf_text(str(LUMEN_PDF))
    assert "LUMEN-5831-ORBIT" in text
    assert "Project Lumen" in text


def test_chunk_text_splits_long_text_with_overlap():
    text = " ".join(f"This is sentence {i} about telemetry." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    assert chunk_text("Short doc.", size=1000, overlap=200) == ["Short doc."]


def test_chunk_text_long_sentence_is_hard_split():
    chunks = chunk_text("x" * 2500, size=1000, overlap=200)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_id_is_stable_and_document_scoped():
    a = chunk_id("docA", "hello world")
    assert a == chunk_id("docA", "hello world")
    assert a != chunk_id("docB", "hello world")
    assert a != chunk_id("docA", "different")


def test_ingest_pdf_indexes_chunks(settings):
    engine = build_engine(settings)
    assert engine.count() == 0
    result = ingest_pdf(engine, "lumen-test", str(LUMEN_PDF), "Project Lumen.pdf")
    assert result["success"] is True
    assert result["chunks"] >= 1
    assert result["added"] >= 1
    assert engine.count() == result["added"]


def test_reindexing_is_idempotent(settings):
    engine = build_engine(settings)
    ingest_pdf(engine, "lumen-test", str(LUMEN_PDF), "Project Lumen.pdf")
    first = engine.count()
    second = ingest_pdf(engine, "lumen-test", str(LUMEN_PDF), "Project Lumen.pdf")
    assert second["added"] == 0
    assert engine.count() == first
