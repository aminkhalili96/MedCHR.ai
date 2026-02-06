from contextlib import contextmanager
from contextvars import ContextVar

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings

_pool: ConnectionPool | None = None
_tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
_actor_id_var: ContextVar[str | None] = ContextVar("actor_id", default=None)


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


@contextmanager
def get_conn():
    global _pool
    settings = get_settings()
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            kwargs={"row_factory": dict_row},
            configure=_configure_connection,
        )
    with _pool.connection() as conn:
        tenant_id = _tenant_id_var.get()
        actor_id = _actor_id_var.get()
        conn.execute("SET app.tenant_id = %s", (tenant_id or "",))
        conn.execute("SET app.actor_id = %s", (actor_id or "",))
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
