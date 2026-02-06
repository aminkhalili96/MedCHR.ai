DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'patients'
      AND column_name = 'dob'
      AND data_type = 'text'
  ) THEN
    EXECUTE $sql$
      ALTER TABLE patients
      ALTER COLUMN dob TYPE DATE
      USING CASE
        WHEN dob IS NULL THEN NULL
        WHEN dob ~ '^\d{4}-\d{2}-\d{2}$' THEN dob::date
        WHEN dob ~ '^\d{2}/\d{2}/\d{4}$' THEN to_date(dob, 'MM/DD/YYYY')
        ELSE NULL
      END
    $sql$;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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

CREATE INDEX IF NOT EXISTS idx_documents_patient_id ON documents(patient_id);
CREATE INDEX IF NOT EXISTS idx_extractions_document_id ON extractions(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_chr_versions_patient_id ON chr_versions(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_patient_id ON audit_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_patient_id ON jobs(patient_id);
CREATE INDEX IF NOT EXISTS idx_jobs_document_id ON jobs(document_id);

DO $$
BEGIN
  BEGIN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw ON embeddings USING hnsw (embedding vector_l2_ops)';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Skipping HNSW index creation; pgvector may not support it.';
  END;
END $$;
