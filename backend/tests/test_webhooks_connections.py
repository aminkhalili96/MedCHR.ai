import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.webhooks import (
    WebhookEvent,
    dispatch_event,
    get_webhooks_for_event,
    register_webhook,
)


def _mock_conn_ctx(conn):
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


def test_register_webhook_uses_context_manager_when_conn_missing():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"id": "w-1"}
    with patch("backend.app.webhooks.get_conn", return_value=_mock_conn_ctx(conn)):
        webhook_id = register_webhook(
            tenant_id="tenant-1",
            url="https://example.test/hook",
            events=[WebhookEvent.CHR_GENERATED.value],
        )

    assert webhook_id == "w-1"
    conn.execute.assert_called_once()
    conn.commit.assert_called_once()


def test_get_webhooks_for_event_uses_context_manager_when_conn_missing():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [{"id": "w-1", "url": "https://example.test", "secret": "s"}]
    with patch("backend.app.webhooks.get_conn", return_value=_mock_conn_ctx(conn)):
        rows = get_webhooks_for_event("tenant-1", WebhookEvent.CHR_GENERATED.value)

    assert rows == [{"id": "w-1", "url": "https://example.test", "secret": "s"}]
    conn.execute.assert_called_once()


def test_dispatch_event_uses_context_manager_when_conn_missing():
    conn = MagicMock()
    with patch("backend.app.webhooks.get_conn", return_value=_mock_conn_ctx(conn)), patch(
        "backend.app.webhooks.get_webhooks_for_event",
        return_value=[{"id": "w-1"}],
    ), patch("backend.app.webhooks.dispatch_webhook", new=AsyncMock(return_value=True)) as dispatch_mock:
        asyncio.run(
            dispatch_event(
                tenant_id="tenant-1",
                event=WebhookEvent.CHR_GENERATED,
                payload={"patient_id": "p-1"},
            )
        )

    dispatch_mock.assert_awaited_once()
    conn.commit.assert_called_once()
