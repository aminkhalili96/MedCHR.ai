# Implementation Plan: Interoperability (Phase 3)

## Objective
Enable data exchange with Electronic Health Records (EHRs) using industry standards (FHIR, LOINC, SNOMED).

## 1. HL7 FHIR R4 Support
**Gap**: No standard API for patient data.

### 1.1 FHIR Data Model Mapping
*   **Library**: `fhir.resources` (Pydantic models for FHIR).
*   **Mapping Strategy**:
    *   `Patient` (MedCHR) -> `Patient` (FHIR)
    *   `Diagnosis` (MedCHR) -> `Condition` (FHIR)
    *   `Medication` (MedCHR) -> `MedicationStatement` (FHIR)
    *   `LabResult` (MedCHR) -> `Observation` (FHIR)
    *   `Document` (MedCHR) -> `DocumentReference` (FHIR)

### 1.2 FHIR API Endpoints
*   Implement a `/fhir/r4` sub-application in FastAPI.
*   **Endpoints**:
    *   `GET /fhir/r4/Patient/{id}`
    *   `GET /fhir/r4/Observation?patient={id}&category=laboratory`
    *   `GET /fhir/r4/Condition?patient={id}`
*   **Response**: JSON-compliant FHIR resources.

## 2. Terminology Services
**Gap**: No standardization of medical terms.

### 2.1 LOINC Implementation (Labs)
*   **Database**: Import core LOINC table (code, long_common_name) into a read-only schema `ref_loinc`.
*   **Auto-Coding**: When extracting labs, use pgvector to search `ref_loinc` embedding vs. extracted test name embedding to suggest LOINC codes.
    *   *Example*: "Fasting Sugar" vector-matches "Glucose [Mass/volume] in Serum or Plasma --fasting".

### 2.2 RxNorm (Medications)
*   **API**: Use NLM RxNorm API (external) or import a simplified RxNorm dataset locally.
*   **Normalization**: Map extracted string "Metformin 500mg" to RxCUI.

## 3. EHR Integration Strategy
*   **SMART on FHIR**: Implement OAuth2 scopes to allow MedCHR to launch *inside* Epic/Cerner.
    *   Uses `fhir-client` libraries.
    *   Requires `launch_context` (patient ID passed from EHR).

## Roadmap Tasks
- [ ] Requirements: Add `fhir.resources`.
- [ ] DB: Create `ref_loinc`, `ref_snomed`, `ref_rxnorm` tables.
- [ ] Script: ETL script to load LOINC csv into Postgres.
- [ ] Backend: Create `app/fhir` router.
- [ ] Backend: Implement mappers (`MedChrToFhirConverter`).
- [ ] AI: Train/Fine-tune embedder for specific medical entity logic (BioBERT or similar).
