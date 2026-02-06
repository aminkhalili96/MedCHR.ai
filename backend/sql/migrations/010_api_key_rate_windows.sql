-- Per-minute API key request windows for rate-limit enforcement.

BEGIN;

CREATE TABLE IF NOT EXISTS api_key_rate_windows (
  api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
  window_start TIMESTAMPTZ NOT NULL,
  request_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (api_key_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_api_key_rate_windows_window_start
  ON api_key_rate_windows (window_start);

COMMIT;
