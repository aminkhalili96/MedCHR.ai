-- Migration 002: Hospital-Grade Foundation
-- Includes Schema for: Security, Clinical, Enterprise, Interop

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. Security & Compliance
-- -----------------------------------------------------------------------------

-- MFA Support
ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_secret TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS backup_codes TEXT[];

-- Advanced Session Management
CREATE TABLE IF NOT EXISTS user_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  ip_address INET,
  user_agent TEXT,
  last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Comprehensive Audit Logging (Immutable)
CREATE TABLE IF NOT EXISTS audit_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  actor_id UUID REFERENCES users(id) ON DELETE SET NULL,
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  ip_address INET,
  resource_type TEXT NOT NULL, -- 'patient', 'document', 'system'
  resource_id UUID,
  action TEXT NOT NULL, -- 'VIEW', 'CREATE', 'UPDATE', 'DELETE', 'EXPORT'
  outcome TEXT NOT NULL, -- 'SUCCESS', 'DENIED', 'FAILURE'
  details JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_time ON audit_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_actor ON audit_events(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_resource ON audit_events(resource_type, resource_id);

-- IP Whitelisting
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS allowed_ips CIDR[];

-- -----------------------------------------------------------------------------
-- 2. Clinical Data (Phase 3/4)
-- -----------------------------------------------------------------------------

-- Allergies
CREATE TABLE IF NOT EXISTS allergies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  substance TEXT NOT NULL,
  reaction TEXT,
  severity TEXT, -- 'mild', 'moderate', 'severe'
  status TEXT DEFAULT 'active', -- 'active', 'resolved'
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_allergies_patient_id ON allergies(patient_id);

-- Vitals
CREATE TABLE IF NOT EXISTS vitals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  recorded_at TIMESTAMPTZ NOT NULL,
  type TEXT NOT NULL, -- 'BP', 'HR', 'Temp', 'Weight', 'SpO2'
  value_1 NUMERIC, -- Systolic or single value
  value_2 NUMERIC, -- Diastolic
  unit TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_vitals_patient_type_time ON vitals(patient_id, type, recorded_at DESC);

-- Immunizations
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
CREATE INDEX IF NOT EXISTS idx_immunizations_patient_id ON immunizations(patient_id);

-- Medications (Enhancement) - Add interaction risk level
ALTER TABLE medications ADD COLUMN IF NOT EXISTS interaction_risk TEXT; -- 'high', 'moderate', 'low'

-- -----------------------------------------------------------------------------
-- 3. Enterprise Scale (Phase 5)
-- -----------------------------------------------------------------------------

-- Hierarchy
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS parent_id UUID REFERENCES tenants(id) ON DELETE SET NULL;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS type TEXT DEFAULT 'clinic'; -- 'organization', 'region', 'clinic'

-- Branding
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS branding_config JSONB DEFAULT '{}'::jsonb;
-- e.g. {"logo_url": "...", "primary_color": "#0055AA"}

-- Tiering
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS tier TEXT DEFAULT 'standard'; -- 'free', 'standard', 'enterprise'

-- -----------------------------------------------------------------------------
-- 4. Interoperability & References
-- -----------------------------------------------------------------------------

-- Reference Tables (LOINC placeholder)
CREATE TABLE IF NOT EXISTS ref_loinc (
  code TEXT PRIMARY KEY,
  long_common_name TEXT,
  component TEXT,
  system TEXT
  -- Intentionally minimal for now
);
-- Enable vector search on LOINC names for mapping
CREATE TABLE IF NOT EXISTS ref_loinc_embeddings (
    code TEXT REFERENCES ref_loinc(code) ON DELETE CASCADE,
    embedding vector(3072),
    PRIMARY KEY (code)
);

COMMIT;
