import json
from typing import List, Dict, Any

from pgvector.psycopg import Vector

from .db import get_conn
from .embeddings import embed_texts


def build_query(structured: dict, notes: str | None = None) -> str:
    parts = [json.dumps(structured, ensure_ascii=False)]
    if notes:
        parts.append(notes)
    return "\n".join(parts)


def retrieve_top_chunks(patient_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    embedding = embed_texts([query])[0]
    vector = Vector(embedding)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                e.chunk_text,
                e.chunk_index,
                e.chunk_start,
                e.chunk_end,
                e.extraction_id,
                d.id as document_id,
                d.filename,
                d.content_type,
                (e.embedding <-> %s) AS distance
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            WHERE d.patient_id = %s
            ORDER BY e.embedding <-> %s
            LIMIT %s
            """,
            (vector, patient_id, vector, top_k),
        ).fetchall()

    return [
        {
            "chunk_text": r["chunk_text"],
            "distance": float(r["distance"]),
            "chunk_index": r.get("chunk_index"),
            "chunk_start": r.get("chunk_start"),
            "chunk_end": r.get("chunk_end"),
            "extraction_id": str(r["extraction_id"]) if r.get("extraction_id") else None,
            "document_id": str(r["document_id"]),
            "filename": r["filename"],
            "content_type": r["content_type"],
        }
        for r in rows
    ]


def retrieve_sparse_chunks(patient_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Sparse retrieval using Postgres full-text search (BM25-like).
    Better for exact terms, dosages, and medical acronyms.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                e.chunk_text,
                e.chunk_index,
                e.chunk_start,
                e.chunk_end,
                e.extraction_id,
                d.id as document_id,
                d.filename,
                d.content_type,
                ts_rank_cd(
                    to_tsvector('english', e.chunk_text),
                    plainto_tsquery('english', %s)
                ) AS rank
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            WHERE d.patient_id = %s
              AND to_tsvector('english', e.chunk_text) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, patient_id, query, top_k),
        ).fetchall()
    
    return [
        {
            "chunk_text": r["chunk_text"],
            "rank": float(r["rank"]),
            "chunk_index": r.get("chunk_index"),
            "chunk_start": r.get("chunk_start"),
            "chunk_end": r.get("chunk_end"),
            "extraction_id": str(r["extraction_id"]) if r.get("extraction_id") else None,
            "document_id": str(r["document_id"]),
            "filename": r["filename"],
            "content_type": r["content_type"],
        }
        for r in rows
    ]


def reciprocal_rank_fusion(
    dense_results: List[Dict], 
    sparse_results: List[Dict], 
    k: int = 60
) -> List[Dict]:
    """
    Reciprocal Rank Fusion combines dense and sparse search results.
    
    RRF score = sum(1 / (k + rank)) for each list where doc appears
    """
    scores = {}
    doc_data = {}
    
    # Score dense results
    for rank, doc in enumerate(dense_results):
        doc_id = (doc["document_id"], doc.get("chunk_index", 0))
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_data[doc_id] = doc
    
    # Score sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = (doc["document_id"], doc.get("chunk_index", 0))
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_data[doc_id] = doc
    
    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    return [
        {**doc_data[doc_id], "rrf_score": scores[doc_id]}
        for doc_id in sorted_ids
        if doc_id in doc_data
    ]


def retrieve_hybrid(patient_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval combining dense embeddings and sparse keyword search.
    
    - Dense: Good for semantic similarity ("kidney problems" matches "renal function")
    - Sparse: Good for exact terms ("metformin 500mg", "K+ 6.2")
    - RRF: Combines both for best results
    """
    # Get more candidates from each method
    dense_results = retrieve_top_chunks(patient_id, query, top_k=top_k * 2)
    sparse_results = retrieve_sparse_chunks(patient_id, query, top_k=top_k * 2)
    
    # Fuse and return top k
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]

