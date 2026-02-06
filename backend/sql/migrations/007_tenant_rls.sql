-- Enforce tenant isolation at the database layer using Row Level Security (RLS).
-- Policies rely on the session GUC `app.tenant_id` set by the application.

BEGIN;

CREATE OR REPLACE FUNCTION current_tenant_uuid()
RETURNS UUID
LANGUAGE SQL
STABLE
AS $$
  SELECT NULLIF(current_setting('app.tenant_id', true), '')::uuid
$$;

-- ---------------------------------------------------------------------------
-- Direct tenant_id tables
-- ---------------------------------------------------------------------------

DO $$
BEGIN
  IF to_regclass('public.tenants') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE tenants ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_tenants_isolation ON tenants';
    EXECUTE 'CREATE POLICY rls_tenants_isolation ON tenants
             USING (id = current_tenant_uuid())
             WITH CHECK (id = current_tenant_uuid())';
  END IF;

  IF to_regclass('public.users') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE users ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_users_tenant ON users';
    EXECUTE 'CREATE POLICY rls_users_tenant ON users
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;

  IF to_regclass('public.patients') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE patients ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_patients_tenant ON patients';
    EXECUTE 'CREATE POLICY rls_patients_tenant ON patients
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;

  IF to_regclass('public.jobs') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE jobs ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_jobs_tenant ON jobs';
    EXECUTE 'CREATE POLICY rls_jobs_tenant ON jobs
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;

  IF to_regclass('public.api_keys') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_api_keys_tenant ON api_keys';
    EXECUTE 'CREATE POLICY rls_api_keys_tenant ON api_keys
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;

  IF to_regclass('public.audit_logs') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_audit_logs_tenant ON audit_logs';
    EXECUTE 'CREATE POLICY rls_audit_logs_tenant ON audit_logs
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;

  IF to_regclass('public.audit_events') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_audit_events_tenant ON audit_events';
    EXECUTE 'CREATE POLICY rls_audit_events_tenant ON audit_events
             USING (tenant_id = current_tenant_uuid())
             WITH CHECK (tenant_id = current_tenant_uuid())';
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- Tables scoped through patient ownership
-- ---------------------------------------------------------------------------

DO $$
BEGIN
  IF to_regclass('public.documents') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE documents ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_documents_patient_tenant ON documents';
    EXECUTE 'CREATE POLICY rls_documents_patient_tenant ON documents
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = documents.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = documents.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.extractions') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE extractions ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_extractions_document_tenant ON extractions';
    EXECUTE 'CREATE POLICY rls_extractions_document_tenant ON extractions
             USING (
               EXISTS (
                 SELECT 1
                 FROM documents d
                 JOIN patients p ON p.id = d.patient_id
                 WHERE d.id = extractions.document_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1
                 FROM documents d
                 JOIN patients p ON p.id = d.patient_id
                 WHERE d.id = extractions.document_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.embeddings') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_embeddings_document_tenant ON embeddings';
    EXECUTE 'CREATE POLICY rls_embeddings_document_tenant ON embeddings
             USING (
               EXISTS (
                 SELECT 1
                 FROM documents d
                 JOIN patients p ON p.id = d.patient_id
                 WHERE d.id = embeddings.document_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1
                 FROM documents d
                 JOIN patients p ON p.id = d.patient_id
                 WHERE d.id = embeddings.document_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.chr_versions') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE chr_versions ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_chr_versions_patient_tenant ON chr_versions';
    EXECUTE 'CREATE POLICY rls_chr_versions_patient_tenant ON chr_versions
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = chr_versions.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = chr_versions.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.medications') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE medications ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_medications_patient_tenant ON medications';
    EXECUTE 'CREATE POLICY rls_medications_patient_tenant ON medications
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = medications.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = medications.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.lab_results') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE lab_results ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_lab_results_patient_tenant ON lab_results';
    EXECUTE 'CREATE POLICY rls_lab_results_patient_tenant ON lab_results
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = lab_results.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = lab_results.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.diagnoses') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE diagnoses ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_diagnoses_patient_tenant ON diagnoses';
    EXECUTE 'CREATE POLICY rls_diagnoses_patient_tenant ON diagnoses
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = diagnoses.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = diagnoses.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.allergies') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE allergies ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_allergies_patient_tenant ON allergies';
    EXECUTE 'CREATE POLICY rls_allergies_patient_tenant ON allergies
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = allergies.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = allergies.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.vitals') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE vitals ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_vitals_patient_tenant ON vitals';
    EXECUTE 'CREATE POLICY rls_vitals_patient_tenant ON vitals
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = vitals.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = vitals.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.immunizations') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE immunizations ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_immunizations_patient_tenant ON immunizations';
    EXECUTE 'CREATE POLICY rls_immunizations_patient_tenant ON immunizations
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = immunizations.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = immunizations.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;

  IF to_regclass('public.patient_events') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE patient_events ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_patient_events_patient_tenant ON patient_events';
    EXECUTE 'CREATE POLICY rls_patient_events_patient_tenant ON patient_events
             USING (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = patient_events.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1 FROM patients p
                 WHERE p.id = patient_events.patient_id
                   AND p.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- Session table scoped through user tenant ownership
-- ---------------------------------------------------------------------------

DO $$
BEGIN
  IF to_regclass('public.ui_sessions') IS NOT NULL THEN
    EXECUTE 'ALTER TABLE ui_sessions ENABLE ROW LEVEL SECURITY';
    EXECUTE 'DROP POLICY IF EXISTS rls_ui_sessions_user_tenant ON ui_sessions';
    EXECUTE 'CREATE POLICY rls_ui_sessions_user_tenant ON ui_sessions
             USING (
               EXISTS (
                 SELECT 1
                 FROM users u
                 WHERE u.id = ui_sessions.user_id
                   AND u.tenant_id = current_tenant_uuid()
               )
             )
             WITH CHECK (
               EXISTS (
                 SELECT 1
                 FROM users u
                 WHERE u.id = ui_sessions.user_id
                   AND u.tenant_id = current_tenant_uuid()
               )
             )';
  END IF;
END $$;

COMMIT;

