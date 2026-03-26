"""Tests for db.py security fixes — SQL injection prevention and pool locking."""

import pytest
from unittest.mock import MagicMock, patch

from backend.app.db import _validate_context_id, get_conn


class TestValidateContextId:
    def test_valid_uuid(self):
        result = _validate_context_id("550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_empty_string(self):
        assert _validate_context_id("") == ""

    def test_rejects_sql_injection(self):
        with pytest.raises(ValueError, match="Invalid context ID"):
            _validate_context_id("'; DROP TABLE users; --")

    def test_rejects_non_uuid_string(self):
        with pytest.raises(ValueError, match="Invalid context ID"):
            _validate_context_id("not-a-uuid")

    def test_rejects_uuid_with_extra_chars(self):
        with pytest.raises(ValueError, match="Invalid context ID"):
            _validate_context_id("550e8400-e29b-41d4-a716-446655440000; DROP TABLE")

    def test_accepts_uppercase_uuid(self):
        result = _validate_context_id("550E8400-E29B-41D4-A716-446655440000")
        assert result == "550E8400-E29B-41D4-A716-446655440000"


class TestGetConnUsesParameterizedSQL:
    @patch("backend.app.db._pool_lock")
    @patch("backend.app.db.get_settings")
    @patch("backend.app.db._tenant_id_var")
    @patch("backend.app.db._actor_id_var")
    def test_set_statements_use_sql_literal(self, mock_actor, mock_tenant, mock_settings, mock_lock):
        """Verify that SET statements use psycopg.sql.Literal, not f-strings."""
        mock_tenant.get.return_value = "550e8400-e29b-41d4-a716-446655440000"
        mock_actor.get.return_value = "660e8400-e29b-41d4-a716-446655440001"

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        import backend.app.db as db_module
        original_pool = db_module._pool
        db_module._pool = mock_pool
        try:
            with get_conn() as conn:
                pass

            # The SET calls should use sql.SQL + sql.Literal, not raw f-strings
            calls = mock_conn.execute.call_args_list
            assert len(calls) >= 2
            for call in calls[:2]:
                arg = call[0][0]
                # psycopg.sql.Composed objects are used with sql.SQL().format()
                assert not isinstance(arg, str), (
                    f"SET statement used a raw string instead of sql.SQL: {arg!r}"
                )
        finally:
            db_module._pool = original_pool
