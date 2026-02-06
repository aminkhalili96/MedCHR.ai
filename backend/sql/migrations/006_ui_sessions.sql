-- Server-side UI sessions.
-- Stores session data in Postgres and keeps only an opaque signed session id in cookies.

CREATE TABLE IF NOT EXISTS ui_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  data JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ui_sessions_user_id ON ui_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_ui_sessions_expires_at ON ui_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_ui_sessions_revoked_at ON ui_sessions(revoked_at);

