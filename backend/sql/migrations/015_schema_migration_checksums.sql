-- Track migration checksums to detect historical migration tampering.

BEGIN;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  checksum TEXT,
  applied_by TEXT
);

ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum TEXT;
ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS applied_by TEXT;

COMMIT;
