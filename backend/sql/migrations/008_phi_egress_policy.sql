-- Tenant PHI egress policy + audit events.
-- Enforces per-tenant AI policy and provides auditability for outbound PHI processing.

BEGIN;

CREATE TABLE IF NOT EXISTS tenant_phi_policies (
  tenant_id UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
  allow_ai_processing BOOLEAN NOT NULL DEFAULT TRUE,
  allowed_processors JSONB NOT NULL DEFAULT '[]'::jsonb,
  require_redaction BOOLEAN NOT NULL DEFAULT FALSE,
  updated_by UUID REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS phi_egress_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  actor_id TEXT,
  processor TEXT NOT NULL,
  operation TEXT NOT NULL,
  model TEXT,
  request_id TEXT,
  allowed BOOLEAN NOT NULL DEFAULT TRUE,
  redaction_applied BOOLEAN NOT NULL DEFAULT FALSE,
  reason TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_phi_egress_events_tenant_time
  ON phi_egress_events (tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_phi_egress_events_processor_time
  ON phi_egress_events (processor, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_phi_egress_events_request_id
  ON phi_egress_events (request_id)
  WHERE request_id IS NOT NULL;

DO $$
BEGIN
  IF to_regclass('public.tenant_phi_policies') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE tenant_phi_policies ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_tenant_phi_policies_tenant ON tenant_phi_policies';
    EXECUTE 'CREATE POLICY rls_tenant_phi_policies_tenant ON tenant_phi_policies
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;

  IF to_regclass('public.phi_egress_events') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE phi_egress_events ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_phi_egress_events_tenant ON phi_egress_events';
    EXECUTE 'CREATE POLICY rls_phi_egress_events_tenant ON phi_egress_events
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;
END $$;

COMMIT;
