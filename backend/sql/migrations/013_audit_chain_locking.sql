-- Make audit hash chaining concurrency-safe by serializing per-tenant chain state.

BEGIN;

CREATE TABLE IF NOT EXISTS audit_event_chain_state (
  tenant_key UUID PRIMARY KEY,
  last_hash TEXT NOT NULL DEFAULT '',
  last_event_id UUID,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION audit_events_set_hash()
RETURNS TRIGGER AS $$
DECLARE
  v_tenant_key UUID;
  v_prev_hash TEXT;
BEGIN
  v_tenant_key := COALESCE(NEW.tenant_id, '00000000-0000-0000-0000-000000000000'::UUID);

  INSERT INTO audit_event_chain_state (tenant_key, last_hash, last_event_id)
  VALUES (v_tenant_key, '', NULL)
  ON CONFLICT (tenant_key) DO NOTHING;

  SELECT last_hash
  INTO v_prev_hash
  FROM audit_event_chain_state
  WHERE tenant_key = v_tenant_key
  FOR UPDATE;

  NEW.prev_hash := COALESCE(v_prev_hash, '');
  NEW.event_hash := encode(
    digest(
      COALESCE(NEW.tenant_id::text, '') || '|' ||
      COALESCE(NEW.actor_id::text, '') || '|' ||
      COALESCE(NEW.actor, '') || '|' ||
      COALESCE(NEW.resource_type, '') || '|' ||
      COALESCE(NEW.resource_id::text, '') || '|' ||
      COALESCE(NEW.action, '') || '|' ||
      COALESCE(NEW.outcome, '') || '|' ||
      COALESCE(NEW.details::text, '{}') || '|' ||
      COALESCE(NEW.request_id, '') || '|' ||
      COALESCE(v_prev_hash, '') || '|' ||
      COALESCE(NEW.event_time::text, ''),
      'sha256'
    ),
    'hex'
  );

  UPDATE audit_event_chain_state
  SET last_hash = NEW.event_hash,
      last_event_id = NEW.id,
      updated_at = NOW()
  WHERE tenant_key = v_tenant_key;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_events_set_hash ON audit_events;
CREATE TRIGGER trg_audit_events_set_hash
  BEFORE INSERT ON audit_events
  FOR EACH ROW
  EXECUTE FUNCTION audit_events_set_hash();

-- Backfill chain state for already-existing events.
WITH latest AS (
  SELECT DISTINCT ON (COALESCE(tenant_id, '00000000-0000-0000-0000-000000000000'::UUID))
    COALESCE(tenant_id, '00000000-0000-0000-0000-000000000000'::UUID) AS tenant_key,
    event_hash,
    id
  FROM audit_events
  WHERE event_hash IS NOT NULL
  ORDER BY COALESCE(tenant_id, '00000000-0000-0000-0000-000000000000'::UUID), event_time DESC, id DESC
)
INSERT INTO audit_event_chain_state (tenant_key, last_hash, last_event_id)
SELECT tenant_key, event_hash, id
FROM latest
ON CONFLICT (tenant_key) DO UPDATE
SET last_hash = EXCLUDED.last_hash,
    last_event_id = EXCLUDED.last_event_id,
    updated_at = NOW();

COMMIT;
