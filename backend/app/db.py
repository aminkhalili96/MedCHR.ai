from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg import connect
from psycopg.rows import dict_row

from .config import get_settings


@contextmanager
def get_conn():
    settings = get_settings()
    conn = connect(settings.database_url, row_factory=dict_row)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()
