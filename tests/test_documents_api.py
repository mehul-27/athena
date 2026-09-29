"""Document upload / listing / deletion / duplicate handling via the API."""

import os


def _upload(client, name, data):
    return client.post("/api/rag/documents", files={"file": (name, data, "application/pdf")})


def test_upload_pdf_indexes_and_lists(client, lumen_bytes):
    resp = _upload(client, "Project Lumen.pdf", lumen_bytes)
    assert resp.status_code == 200
    doc = resp.json()["document"]
    assert doc["filename"] == "Project Lumen.pdf"
    assert doc["chunks"] >= 1

    listing = client.get("/api/rag/documents").json()
    assert len(listing["documents"]) == 1
    assert client.app.state.rag_engine.count() >= 1


def test_upload_stores_file_on_disk(client, lumen_bytes, settings):
    _upload(client, "Project Lumen.pdf", lumen_bytes)
    files = os.listdir(settings.resolved_documents_dir())
    assert any(f.endswith("Project Lumen.pdf") for f in files)


def test_upload_rejects_non_pdf(client):
    resp = _upload(client, "notes.txt", b"hello, not a pdf")
    assert resp.status_code == 400
    assert client.app.state.rag_engine.count() == 0


def test_duplicate_upload_is_not_reindexed(client, lumen_bytes):
    first = _upload(client, "Project Lumen.pdf", lumen_bytes).json()
    count_after_first = client.app.state.rag_engine.count()

    second = _upload(client, "Project Lumen.pdf", lumen_bytes).json()
    assert second["duplicate"] is True
    assert second["document"]["document_id"] == first["document"]["document_id"]
    assert client.app.state.rag_engine.count() == count_after_first
    assert len(client.get("/api/rag/documents").json()["documents"]) == 1


def test_delete_removes_file_and_vectors(client, lumen_bytes, settings):
    doc = _upload(client, "Project Lumen.pdf", lumen_bytes).json()["document"]
    assert client.app.state.rag_engine.count() >= 1

    resp = client.delete(f"/api/rag/documents/{doc['document_id']}")
    assert resp.status_code == 200
    assert resp.json()["removed_chunks"] >= 1

    assert client.app.state.rag_engine.count() == 0, "no orphan vectors may remain"
    assert client.get("/api/rag/documents").json()["documents"] == []
    assert not os.path.exists(doc["stored_path"])


def test_delete_unknown_document_404(client):
    assert client.delete("/api/rag/documents/does-not-exist").status_code == 404
