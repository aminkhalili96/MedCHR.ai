-- Store MFA secrets encrypted at rest.

BEGIN;

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS mfa_secret_encrypted TEXT;

CREATE INDEX IF NOT EXISTS idx_users_mfa_enabled
  ON users (mfa_enabled)
  WHERE mfa_enabled = TRUE;

COMMIT;
