-- 017_clinical_sections.sql
-- Adds tables for comprehensive CHR report sections:
-- encounters, clinical_history, family_history, treatment_plans, review_of_systems

-- Clinical Encounters (links documents/reports to specific visits)
CREATE TABLE IF NOT EXISTS encounters (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    tenant_id       UUID NOT NULL,
    encounter_date  DATE NOT NULL DEFAULT CURRENT_DATE,
    encounter_type  TEXT NOT NULL DEFAULT 'outpatient',  -- outpatient, inpatient, emergency, telehealth
    chief_complaint TEXT,
    hpi             TEXT,  -- History of Present Illness
    provider_name   TEXT,
    provider_role   TEXT,
    status          TEXT NOT NULL DEFAULT 'active',  -- active, completed, cancelled
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Clinical History (PMH, surgical, hospitalizations)
CREATE TABLE IF NOT EXISTS clinical_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    tenant_id       UUID NOT NULL,
    history_type    TEXT NOT NULL,  -- medical, surgical, hospitalization
    condition       TEXT NOT NULL,
    icd_code        TEXT,
    date_onset      DATE,
    date_resolved   DATE,
    status          TEXT NOT NULL DEFAULT 'historical',  -- active, resolved, historical
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Family History
CREATE TABLE IF NOT EXISTS family_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    tenant_id       UUID NOT NULL,
    relation        TEXT NOT NULL,  -- mother, father, sibling, maternal_grandmother, etc.
    condition       TEXT NOT NULL,
    age_at_onset    INT,
    age_at_death    INT,
    is_deceased     BOOLEAN DEFAULT FALSE,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Treatment Plans / Care Plans
CREATE TABLE IF NOT EXISTS treatment_plans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    encounter_id    UUID REFERENCES encounters(id),
    tenant_id       UUID NOT NULL,
    plan_type       TEXT NOT NULL DEFAULT 'general',  -- general, medication, procedure, referral, follow_up
    description     TEXT NOT NULL,
    priority        TEXT DEFAULT 'routine',  -- stat, urgent, routine
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, in_progress, completed, cancelled
    target_date     DATE,
    completed_at    TIMESTAMPTZ,
    created_by      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Review of Systems (ROS) - per encounter
CREATE TABLE IF NOT EXISTS review_of_systems (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    encounter_id    UUID REFERENCES encounters(id),
    tenant_id       UUID NOT NULL,
    system_name     TEXT NOT NULL,  -- constitutional, heent, cardiovascular, respiratory, gi, gu, msk, neuro, psych, skin, endo, heme_lymph
    findings        TEXT NOT NULL DEFAULT 'negative',
    is_positive     BOOLEAN DEFAULT FALSE,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add gender and contact fields to patients if not present
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'gender') THEN
        ALTER TABLE patients ADD COLUMN gender TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'phone') THEN
        ALTER TABLE patients ADD COLUMN phone TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'email') THEN
        ALTER TABLE patients ADD COLUMN email TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'emergency_contact') THEN
        ALTER TABLE patients ADD COLUMN emergency_contact JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'insurance') THEN
        ALTER TABLE patients ADD COLUMN insurance JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'social_history') THEN
        ALTER TABLE patients ADD COLUMN social_history JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'past_medical_history') THEN
        ALTER TABLE patients ADD COLUMN past_medical_history JSONB DEFAULT '[]';
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_encounters_patient ON encounters(patient_id);
CREATE INDEX IF NOT EXISTS idx_encounters_tenant ON encounters(tenant_id);
CREATE INDEX IF NOT EXISTS idx_clinical_history_patient ON clinical_history(patient_id);
CREATE INDEX IF NOT EXISTS idx_family_history_patient ON family_history(patient_id);
CREATE INDEX IF NOT EXISTS idx_treatment_plans_patient ON treatment_plans(patient_id);
CREATE INDEX IF NOT EXISTS idx_treatment_plans_encounter ON treatment_plans(encounter_id);
CREATE INDEX IF NOT EXISTS idx_ros_patient ON review_of_systems(patient_id);
CREATE INDEX IF NOT EXISTS idx_ros_encounter ON review_of_systems(encounter_id);

-- RLS policies for new tables (if tenant RLS is active)
DO $$
BEGIN
    -- encounters
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'encounters' AND policyname = 'tenant_isolation_encounters') THEN
        EXECUTE 'ALTER TABLE encounters ENABLE ROW LEVEL SECURITY';
        EXECUTE 'CREATE POLICY tenant_isolation_encounters ON encounters USING (tenant_id::text = current_setting(''app.tenant_id'', true))';
    END IF;
    -- clinical_history
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'clinical_history' AND policyname = 'tenant_isolation_clinical_history') THEN
        EXECUTE 'ALTER TABLE clinical_history ENABLE ROW LEVEL SECURITY';
        EXECUTE 'CREATE POLICY tenant_isolation_clinical_history ON clinical_history USING (tenant_id::text = current_setting(''app.tenant_id'', true))';
    END IF;
    -- family_history
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'family_history' AND policyname = 'tenant_isolation_family_history') THEN
        EXECUTE 'ALTER TABLE family_history ENABLE ROW LEVEL SECURITY';
        EXECUTE 'CREATE POLICY tenant_isolation_family_history ON family_history USING (tenant_id::text = current_setting(''app.tenant_id'', true))';
    END IF;
    -- treatment_plans
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'treatment_plans' AND policyname = 'tenant_isolation_treatment_plans') THEN
        EXECUTE 'ALTER TABLE treatment_plans ENABLE ROW LEVEL SECURITY';
        EXECUTE 'CREATE POLICY tenant_isolation_treatment_plans ON treatment_plans USING (tenant_id::text = current_setting(''app.tenant_id'', true))';
    END IF;
    -- review_of_systems
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'review_of_systems' AND policyname = 'tenant_isolation_ros') THEN
        EXECUTE 'ALTER TABLE review_of_systems ENABLE ROW LEVEL SECURITY';
        EXECUTE 'CREATE POLICY tenant_isolation_ros ON review_of_systems USING (tenant_id::text = current_setting(''app.tenant_id'', true))';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'RLS policy creation skipped: %', SQLERRM;
END $$;
