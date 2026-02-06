-- Hospital Grade Schema Extensions (v3)
-- Additional tables for new modules

-- ============================================================================
-- Document Signatures (S05)
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_signatures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    signer_id UUID NOT NULL REFERENCES users(id),
    signer_name VARCHAR(255) NOT NULL,
    signer_role VARCHAR(50) NOT NULL,
    signature_type VARCHAR(20) DEFAULT 'primary' CHECK (signature_type IN ('primary', 'cosign', 'amendment')),
    signature TEXT NOT NULL,
    signed_at TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    revoke_reason TEXT
);

CREATE INDEX idx_doc_signatures_document ON document_signatures(document_id);
CREATE INDEX idx_doc_signatures_signer ON document_signatures(signer_id);

-- ============================================================================
-- Data Corrections Tracking (DQ04)
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    field VARCHAR(100) NOT NULL,
    original_value JSONB,
    corrected_value JSONB,
    corrected_by UUID REFERENCES users(id),
    corrected_at TIMESTAMPTZ DEFAULT NOW(),
    reason TEXT
);

CREATE INDEX idx_corrections_entity ON data_corrections(entity_type, entity_id);
CREATE INDEX idx_corrections_user ON data_corrections(corrected_by);

-- ============================================================================
-- LOINC Embeddings for Vector Search (T02)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ref_loinc_embeddings (
    code VARCHAR(20) PRIMARY KEY REFERENCES ref_loinc(code),
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_loinc_embedding ON ref_loinc_embeddings USING ivfflat (embedding vector_cosine_ops);

-- ============================================================================
-- SSO Sessions (S02)
-- ============================================================================

ALTER TABLE users ADD COLUMN IF NOT EXISTS sso_provider VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS sso_sub VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_users_sso ON users(sso_provider, sso_sub);

-- ============================================================================
-- Tenant Domains for SSO JIT Provisioning
-- ============================================================================

ALTER TABLE tenants ADD COLUMN IF NOT EXISTS domain VARCHAR(255);
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS sso_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS sso_config JSONB;

CREATE UNIQUE INDEX IF NOT EXISTS idx_tenant_domain ON tenants(domain) WHERE domain IS NOT NULL;

-- ============================================================================
-- CHR Version Signing Support
-- ============================================================================

ALTER TABLE chr_versions ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'draft';
ALTER TABLE chr_versions ADD COLUMN IF NOT EXISTS cosigner_required UUID;
ALTER TABLE chr_versions ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);

-- ============================================================================
-- Metrics and Telemetry (M06)
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(36),
    user_id UUID,
    tenant_id UUID,
    model VARCHAR(50),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd DECIMAL(10, 6),
    duration_ms INTEGER,
    endpoint VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_llm_usage_tenant ON llm_usage(tenant_id, created_at);
CREATE INDEX idx_llm_usage_model ON llm_usage(model, created_at);

-- ============================================================================
-- Alert History (C01-C05)
-- ============================================================================

CREATE TABLE IF NOT EXISTS clinical_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    acknowledged_by UUID REFERENCES users(id),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_clinical_alerts_patient ON clinical_alerts(patient_id, created_at DESC);
CREATE INDEX idx_clinical_alerts_severity ON clinical_alerts(severity) WHERE acknowledged_at IS NULL;

-- ============================================================================
-- Document Classification Metadata (D01)
-- ============================================================================

ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type VARCHAR(50);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS classification_confidence REAL;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS report_date DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS provider_name VARCHAR(255);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS facility_name VARCHAR(255);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS quality_score REAL;

-- ============================================================================
-- Extraction Quality Metrics
-- ============================================================================

ALTER TABLE extractions ADD COLUMN IF NOT EXISTS overall_confidence REAL;
ALTER TABLE extractions ADD COLUMN IF NOT EXISTS quality_score REAL;
ALTER TABLE extractions ADD COLUMN IF NOT EXISTS issues JSONB;
ALTER TABLE extractions ADD COLUMN IF NOT EXISTS safety_alerts JSONB;

-- ============================================================================
-- Full-Text Search Index for Hybrid RAG (A08)
-- ============================================================================

-- Add GIN index for full-text search on embeddings chunk_text
CREATE INDEX IF NOT EXISTS idx_embeddings_fts ON embeddings USING gin(to_tsvector('english', chunk_text));

-- ============================================================================
-- Report Templates (R07)
-- ============================================================================

CREATE TABLE IF NOT EXISTS report_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    name VARCHAR(100) NOT NULL,
    display_name VARCHAR(255),
    specialty VARCHAR(50),
    sections JSONB NOT NULL,
    header_template TEXT,
    footer_template TEXT,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_templates_tenant ON report_templates(tenant_id);
CREATE INDEX idx_templates_specialty ON report_templates(specialty);

-- ============================================================================
-- API Keys with Scopes (F05)
-- ============================================================================

ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scopes JSONB DEFAULT '["read"]'::jsonb;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rate_limit INTEGER DEFAULT 1000;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS request_count INTEGER DEFAULT 0;

-- ============================================================================
-- Consent Management (S10)
-- ============================================================================

CREATE TABLE IF NOT EXISTS patient_consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    consent_type VARCHAR(50) NOT NULL,
    granted BOOLEAN NOT NULL,
    granted_by VARCHAR(255),
    granted_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    document_url TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(patient_id, consent_type)
);

CREATE INDEX idx_consents_patient ON patient_consents(patient_id);

-- ============================================================================
-- IP Whitelist Enforcement (S04)
-- ============================================================================

-- Already exists: ALTER TABLE tenants ADD COLUMN ip_whitelist JSONB;

-- Add enforcement trigger or check constraint
CREATE OR REPLACE FUNCTION check_ip_whitelist(tenant_uuid UUID, client_ip TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    whitelist JSONB;
BEGIN
    SELECT ip_whitelist INTO whitelist FROM tenants WHERE id = tenant_uuid;
    
    IF whitelist IS NULL OR jsonb_array_length(whitelist) = 0 THEN
        RETURN TRUE;  -- No whitelist = allow all
    END IF;
    
    RETURN whitelist ? client_ip;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Grant permissions
-- ============================================================================

-- These would be run by the admin user
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO medchr_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO medchr_app;
