-- Store MFA setup secrets server-side (avoid leaking in cookie sessions).
-- These records are short-lived and should be periodically purged.

CREATE TABLE IF NOT EXISTS mfa_setup_tokens (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  secret TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  used_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_mfa_setup_tokens_active_user
  ON mfa_setup_tokens(user_id)
  WHERE used_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_mfa_setup_tokens_user_id ON mfa_setup_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_mfa_setup_tokens_tenant_id ON mfa_setup_tokens(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mfa_setup_tokens_expires_at ON mfa_setup_tokens(expires_at);

