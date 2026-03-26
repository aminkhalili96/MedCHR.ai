import logging
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()
_tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
_actor_id_var: ContextVar[str | None] = ContextVar("actor_id", default=None)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_logger = logging.getLogger(__name__)


def get_tenant_context() -> str | None:
    return _tenant_id_var.get()


def get_actor_context() -> str | None:
    return _actor_id_var.get()


def set_tenant_context(tenant_id: str | None) -> None:
    _tenant_id_var.set(tenant_id)


def clear_tenant_context() -> None:
    _tenant_id_var.set(None)
    _actor_id_var.set(None)


def set_actor_context(actor_id: str | None) -> None:
    _actor_id_var.set(actor_id)


def _configure_connection(conn) -> None:
    settings = get_settings()
    register_vector(conn)
    timeout_ms = int(settings.db_statement_timeout_ms)
    conn.execute(f"SET statement_timeout = {timeout_ms}")
    conn.commit()


def _validate_context_id(value: str) -> str:
    """Validate that a context ID is a valid UUID or empty string."""
    if not value:
        return ""
    if not _UUID_RE.match(value):
        raise ValueError(f"Invalid context ID (expected UUID): {value!r}")
    return value


@contextmanager
def get_conn():
    global _pool
    settings = get_settings()
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                kwargs={"row_factory": dict_row},
                configure=_configure_connection,
            )
    with _pool.connection() as conn:
        tenant_id = _validate_context_id(_tenant_id_var.get() or "")
        actor_id = _validate_context_id(_actor_id_var.get() or "")
        conn.execute(
            sql.SQL("SET app.tenant_id = {}").format(sql.Literal(tenant_id))
        )
        conn.execute(
            sql.SQL("SET app.actor_id = {}").format(sql.Literal(actor_id))
        )
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
