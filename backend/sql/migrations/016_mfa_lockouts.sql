-- Persist MFA lockout state by user/flow to prevent bypass via session reset.

BEGIN;

CREATE TABLE IF NOT EXISTS mfa_lockouts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  flow TEXT NOT NULL CHECK (flow IN ('login', 'step_up')),
  failed_attempts INTEGER NOT NULL DEFAULT 0,
  lockout_until TIMESTAMPTZ,
  last_failed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, flow)
);

CREATE INDEX IF NOT EXISTS idx_mfa_lockouts_lockout_until
  ON mfa_lockouts (lockout_until)
  WHERE lockout_until IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mfa_lockouts_user_flow
  ON mfa_lockouts (user_id, flow);

COMMIT;
