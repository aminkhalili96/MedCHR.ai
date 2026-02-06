-- Harden audit_events for append-only immutable logging.

BEGIN;

ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS actor TEXT;
ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS user_agent TEXT;
ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS request_id TEXT;
ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS prev_hash TEXT;
ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS event_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_audit_events_request_id
  ON audit_events (request_id)
  WHERE request_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_audit_events_event_hash
  ON audit_events (event_hash)
  WHERE event_hash IS NOT NULL;

CREATE OR REPLACE FUNCTION audit_events_set_hash()
RETURNS TRIGGER AS $$
DECLARE
  last_hash TEXT;
BEGIN
  SELECT event_hash
  INTO last_hash
  FROM audit_events
  WHERE tenant_id IS NOT DISTINCT FROM NEW.tenant_id
  ORDER BY event_time DESC, id DESC
  LIMIT 1;

  NEW.prev_hash := COALESCE(last_hash, '');
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
      COALESCE(last_hash, '') || '|' ||
      COALESCE(NEW.event_time::text, ''),
      'sha256'
    ),
    'hex'
  );

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_events_set_hash ON audit_events;
CREATE TRIGGER trg_audit_events_set_hash
  BEFORE INSERT ON audit_events
  FOR EACH ROW
  EXECUTE FUNCTION audit_events_set_hash();

CREATE OR REPLACE FUNCTION audit_events_block_mutation()
RETURNS TRIGGER AS $$
BEGIN
  RAISE EXCEPTION 'audit_events is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_events_no_update ON audit_events;
CREATE TRIGGER trg_audit_events_no_update
  BEFORE UPDATE ON audit_events
  FOR EACH ROW
  EXECUTE FUNCTION audit_events_block_mutation();

DROP TRIGGER IF EXISTS trg_audit_events_no_delete ON audit_events;
CREATE TRIGGER trg_audit_events_no_delete
  BEFORE DELETE ON audit_events
  FOR EACH ROW
  EXECUTE FUNCTION audit_events_block_mutation();

COMMIT;
