import json
from typing import List, Tuple

from pgvector.psycopg import Vector

from .db import get_conn
from .embeddings import embed_texts


def build_query(structured: dict, notes: str | None = None) -> str:
    parts = [json.dumps(structured, ensure_ascii=False)]
    if notes:
        parts.append(notes)
    return "\n".join(parts)


def retrieve_top_chunks(patient_id: str, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    embedding = embed_texts([query])[0]
    vector = Vector(embedding)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT e.chunk_text, (e.embedding <-> %s) AS distance
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            WHERE d.patient_id = %s
            ORDER BY e.embedding <-> %s
            LIMIT %s
            """,
            (vector, patient_id, vector, top_k),
        ).fetchall()

    return [(r["chunk_text"], float(r["distance"])) for r in rows]
