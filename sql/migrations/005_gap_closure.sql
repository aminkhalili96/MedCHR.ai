-- Migration: Gap Closure Features
-- Adds columns and tables for: Trends, Provenance, Timeline, Normalization

-- 1. Add service_date to extractions for Longitudinal Trends
ALTER TABLE extractions ADD COLUMN IF NOT EXISTS service_date DATE;

-- 2. Add source tracking for Biomarker Provenance (already have document_id, this is for display)
-- No schema change needed, but ensure structured JSON includes source_filename

-- 3. Create lab_aliases table for Semantic Normalization
CREATE TABLE IF NOT EXISTS lab_aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alias TEXT NOT NULL UNIQUE,
    canonical_name TEXT NOT NULL,
    loinc_code TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Create patient_events table for Timeline
CREATE TABLE IF NOT EXISTS patient_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL, -- 'lab', 'visit', 'hospitalization', 'procedure', 'diagnosis'
    event_date DATE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    source_document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patient_events_patient ON patient_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_events_date ON patient_events(event_date);

-- 5. Create suggested_diagnoses table for Diagnosis Suggestions
CREATE TABLE IF NOT EXISTS suggested_diagnoses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    diagnosis_text TEXT NOT NULL,
    icd10_code TEXT,
    confidence TEXT, -- 'high', 'medium', 'low'
    status TEXT DEFAULT 'pending', -- 'pending', 'accepted', 'rejected'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_suggested_diagnoses_patient ON suggested_diagnoses(patient_id);

-- 6. Add genetics_interpretation column to patients for cached interpretations
ALTER TABLE patients ADD COLUMN IF NOT EXISTS genetics_interpretation JSONB;
