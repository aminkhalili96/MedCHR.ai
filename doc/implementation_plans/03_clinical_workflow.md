# Implementation Plan: Clinical Workflow (Phase 4)

## Objective
Deepen clinical data capture to support actual medical decision-making (Allergies, Vitals, Problems).

## 1. Expanded Clinical Data Model
**Gap**: Missing critical health domains.

### 1.1 New Schema Entities
```sql
CREATE TABLE allergies (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  substance TEXT NOT NULL, -- e.g. "Penicillin"
  reaction TEXT, -- "Anaphylaxis"
  severity TEXT, -- "High", "Moderate", "Low"
  status TEXT DEFAULT 'active'
);

CREATE TABLE vitals (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  recorded_at TIMESTAMPTZ NOT NULL,
  type TEXT NOT NULL, -- "BP", "HR", "Temp", "Weight"
  value_1 NUMERIC, -- Systolic / Value
  value_2 NUMERIC, -- Diastolic
  unit TEXT
);

CREATE TABLE immunizations (
  id UUID PRIMARY KEY,
  patient_id UUID REFERENCES patients(id),
  vaccine_name TEXT,
  date_administered DATE,
  lot_number TEXT
);
```

## 2. Medication Management
**Gap**: Simple list vs. Safe Prescribing support.

*   **Interaction Checking**: Integrate with an API (e.g., NLM interaction API or proprietary drug database) to check current med list.
*   **Reconciliation UI**:
    *   Split view: "Extracted from Notes" vs "Current Active List".
    *   Drag-and-drop workflow to confirm meds.

## 3. Problem List Management
*   **Evolution**: Convert simple "Diagnoses" to a robust "Problem List".
*   **Fields**: Add `clinical_status` (active, recurrence, relapse, remission) and `verification_status` (unconfirmed, provisional, differential, confirmed).

## 4. Care Gaps & Reminders
*   **Logic Engine**: Simple rule engine to scan patient data.
    *   *Rule*: `If Age > 50 AND Gender = Female AND No Mammogram in > 2 years -> Flag Care Gap`.
*   **Implementation**: Background job `calculators.py` running nightly.

## Roadmap Tasks
- [ ] DB Migration: Create `allergies`, `vitals`, `immunizations` tables.
- [ ] Backend: CRUD endpoints for new entities.
- [ ] Extraction: Update `extract.py` prompts to specifically look for allergy sections and vital signs grids.
- [ ] Frontend: Create "Clinical Summary" dashboard widget showing Vitals trends (Chart.js).
