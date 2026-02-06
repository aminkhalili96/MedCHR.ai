-- Force RLS on tenant-scoped tables to prevent owner-role bypass.

BEGIN;

DO $$
DECLARE
  table_name TEXT;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'tenants',
    'users',
    'patients',
    'documents',
    'extractions',
    'embeddings',
    'chr_versions',
    'medications',
    'lab_results',
    'diagnoses',
    'allergies',
    'vitals',
    'immunizations',
    'patient_events',
    'jobs',
    'api_keys',
    'audit_logs',
    'audit_events',
    'ui_sessions',
    'tenant_phi_policies',
    'phi_egress_events'
  ] LOOP
    IF to_regclass('public.' || table_name) IS NOT NULL THEN
      EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
      EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
    END IF;
  END LOOP;
END $$;

COMMIT;
