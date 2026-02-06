-- Bootstrap migration
-- Purpose: Make fresh deployments reproducible by creating the baseline schema
-- expected by the application code and later migrations.
--
-- Notes:
-- - This migration is intentionally idempotent.
-- - It also backfills tenant_id for older single-tenant dev schemas.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Tenancy
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  -- JSON array of allowed IPs/CIDRs (used by ip_whitelist module + v3 helper fn)
  ip_whitelist JSONB DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Default/system tenant for local development + backfills
INSERT INTO tenants (id, name)
VALUES ('00000000-0000-0000-0000-000000000000', 'System Tenant')
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Users (UI auth)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'clinician', -- 'admin', 'clinician'
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  mfa_enabled BOOLEAN DEFAULT FALSE,
  mfa_secret TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Backfill older schemas that lacked tenant_id
ALTER TABLE users ADD COLUMN IF NOT EXISTS tenant_id UUID;
UPDATE users SET tenant_id = '00000000-0000-0000-0000-000000000000' WHERE tenant_id IS NULL;
ALTER TABLE users ALTER COLUMN tenant_id SET NOT NULL;

-- ---------------------------------------------------------------------------
-- Patients + Documents
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS patients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  full_name TEXT NOT NULL,
  dob DATE,
  notes TEXT,
  lifestyle JSONB DEFAULT '{}'::jsonb,
  genetics JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE patients ADD COLUMN IF NOT EXISTS tenant_id UUID;
UPDATE patients SET tenant_id = '00000000-0000-0000-0000-000000000000' WHERE tenant_id IS NULL;
ALTER TABLE patients ALTER COLUMN tenant_id SET NOT NULL;

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  content_type TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Extractions + Embeddings + CHR
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS extractions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  raw_text TEXT,
  structured JSONB,
  -- Used by /api/gap features and timeline ordering
  service_date DATE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE extractions ADD COLUMN IF NOT EXISTS service_date DATE;

-- text-embedding-3-large uses 3072 dimensions
CREATE TABLE IF NOT EXISTS embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  chunk_index INTEGER,
  chunk_start INTEGER,
  chunk_end INTEGER,
  chunk_text TEXT NOT NULL,
  embedding vector(3072) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chr_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  draft JSONB NOT NULL,
  report_edits JSONB DEFAULT '{}'::jsonb,
  status TEXT NOT NULL DEFAULT 'draft',
  finalized_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS edits (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chr_version_id UUID NOT NULL REFERENCES chr_versions(id) ON DELETE CASCADE,
  field TEXT,
  old_value TEXT,
  new_value TEXT,
  editor TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Audit logs (MVP; enterprise audit_events added in later migrations)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  patient_id UUID REFERENCES patients(id) ON DELETE SET NULL,
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  actor TEXT,
  action TEXT NOT NULL,
  details JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS tenant_id UUID;

-- ---------------------------------------------------------------------------
-- Background Jobs
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  payload JSONB DEFAULT '{}'::jsonb,
  patient_id UUID REFERENCES patients(id) ON DELETE SET NULL,
  document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS tenant_id UUID;

-- ---------------------------------------------------------------------------
-- Structured clinical tables (required by extraction pipeline)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS medications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  dosage TEXT,
  frequency TEXT,
  route TEXT,
  start_date DATE,
  end_date DATE,
  status TEXT DEFAULT 'active',
  interaction_risk TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lab_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  test_name TEXT NOT NULL,
  value TEXT NOT NULL,
  unit TEXT,
  flag TEXT,
  reference_range TEXT,
  test_date DATE,
  panel TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS diagnoses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  condition TEXT NOT NULL,
  code TEXT,
  status TEXT,
  date_onset DATE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Additional clinical tables used by clinical router
CREATE TABLE IF NOT EXISTS allergies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  substance TEXT NOT NULL,
  reaction TEXT,
  severity TEXT,
  status TEXT DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vitals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  recorded_at TIMESTAMPTZ NOT NULL,
  type TEXT NOT NULL,
  value_1 NUMERIC,
  value_2 NUMERIC,
  unit TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS immunizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  vaccine_name TEXT NOT NULL,
  date_administered DATE,
  lot_number TEXT,
  site TEXT,
  status TEXT DEFAULT 'completed',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- GAP feature: timeline events table (optional, but referenced by API)
CREATE TABLE IF NOT EXISTS patient_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  event_date DATE NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  source_document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- API keys (referenced by later migrations; not yet wired into auth)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  name TEXT,
  key_hash TEXT UNIQUE,
  scopes JSONB DEFAULT '["read"]'::jsonb,
  rate_limit INTEGER DEFAULT 1000,
  last_used_at TIMESTAMPTZ,
  request_count INTEGER DEFAULT 0,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes (minimal baseline; additional indexes added in later migrations)
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_patients_tenant_id ON patients(tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_patient_id ON documents(patient_id);
CREATE INDEX IF NOT EXISTS idx_extractions_document_id ON extractions(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_chr_versions_patient_id ON chr_versions(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_patient_id ON audit_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_patient_id ON jobs(patient_id);
CREATE INDEX IF NOT EXISTS idx_jobs_document_id ON jobs(document_id);

