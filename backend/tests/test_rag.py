from unittest.mock import MagicMock, patch

from backend.app.rag import retrieve_top_chunks, DEFAULT_MIN_SIMILARITY


def test_retrieve_top_chunks_filters_by_embedding_dimension():
    with patch("backend.app.rag.embed_texts", return_value=[[0.1, 0.2, 0.3]]), patch(
        "backend.app.rag.get_conn"
    ) as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []

        result = retrieve_top_chunks("patient-1", "renal labs", top_k=5)

        assert result == []
        sql, params = mock_conn.execute.call_args[0]
        assert "vector_dims(e.embedding) = %s" in sql
        assert "ORDER BY distance" in sql
        assert params[1] == "patient-1"
        assert params[2] == 3


def test_retrieve_top_chunks_maps_row_payload():
    row = {
        "chunk_text": "Creatinine elevated",
        "distance": 0.12,
        "chunk_index": 4,
        "chunk_start": 100,
        "chunk_end": 140,
        "extraction_id": "extract-1",
        "document_id": "doc-1",
        "filename": "lab_report.pdf",
        "content_type": "application/pdf",
    }

    with patch("backend.app.rag.embed_texts", return_value=[[0.1, 0.2]]), patch(
        "backend.app.rag.get_conn"
    ) as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [row]

        result = retrieve_top_chunks("patient-1", "creatinine")

    assert len(result) == 1
    assert result[0]["chunk_text"] == "Creatinine elevated"
    assert result[0]["distance"] == 0.12
    assert result[0]["document_id"] == "doc-1"


def test_retrieve_top_chunks_filters_by_min_similarity():
    """Chunks with distance > min_similarity should be excluded."""
    close_row = {
        "chunk_text": "Relevant result",
        "distance": 0.3,
        "chunk_index": 0,
        "chunk_start": 0,
        "chunk_end": 50,
        "extraction_id": "e-1",
        "document_id": "doc-1",
        "filename": "report.pdf",
        "content_type": "application/pdf",
    }
    far_row = {
        "chunk_text": "Irrelevant result",
        "distance": 0.95,
        "chunk_index": 1,
        "chunk_start": 50,
        "chunk_end": 100,
        "extraction_id": "e-2",
        "document_id": "doc-2",
        "filename": "unrelated.pdf",
        "content_type": "application/pdf",
    }

    with patch("backend.app.rag.embed_texts", return_value=[[0.1, 0.2]]), patch(
        "backend.app.rag.get_conn"
    ) as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [close_row, far_row]

        result = retrieve_top_chunks("patient-1", "query", min_similarity=0.75)

    assert len(result) == 1
    assert result[0]["chunk_text"] == "Relevant result"


def test_default_min_similarity_is_reasonable():
    assert 0.5 <= DEFAULT_MIN_SIMILARITY <= 1.0
