-- Retention export manifest registry for immutable evidence tracking.

BEGIN;

CREATE TABLE IF NOT EXISTS retention_manifests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  table_name TEXT NOT NULL,
  retention_days INTEGER NOT NULL,
  row_count INTEGER NOT NULL DEFAULT 0,
  export_uri TEXT NOT NULL,
  checksum TEXT NOT NULL,
  immutable_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
  manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retention_manifests_table_time
  ON retention_manifests (table_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_retention_manifests_immutable
  ON retention_manifests (immutable_confirmed, created_at DESC);

COMMIT;
