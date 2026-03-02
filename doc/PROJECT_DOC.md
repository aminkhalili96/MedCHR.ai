# MedCHR Project Doc

## Overview
MedCHR is a clinician-in-the-loop platform that turns raw patient records into editable Client Health Reports (CHRs). The goal is to help licensed clinicians synthesize labs, diagnoses, procedures, meds, and genetics into a structured, reviewable report with citations back to the source data. The system supports upload, extraction, normalization, AI drafting, clinician edits, and final sign-off before sharing with patients.

## Objectives
- Reduce time spent summarizing patient data while preserving clinical control.
- Produce structured CHRs with citations to source documents.
- Support multi-patient workflows, versioning, and audit trails.
- Demonstrate safe AI usage: human review, editability, and explicit sign-off.
- **Ensure strictly isolated multi-tenant data security.**

## Target Users
- Licensed clinicians in outpatient, integrative, or functional medicine.
- Clinical teams that need summary reports for patient consults.

## Core Features (MVP)
- Patient and document management (upload PDFs/images/notes).
- Document parsing and structured extraction (labs, diagnoses, procedures, meds, genetics).
- Biomarker normalization (units, reference ranges, abnormal flags).
- RAG retrieval with pgvector (top-K chunks for drafting).
- CHR drafting with source citations.
- Clinician edit UI with change tracking and final sign-off.
- **Multi-user Role-Based Access Control (RBAC) with tenant isolation.**
- **Admin Dashboard for user and tenant management.**
- Auth-gated clinician UI and basic audit log.
- RAG top-K viewer and embeddings debug UI (dev tools).
- Export finalized CHR (PDF/HTML).
- **Clinical Analytics (Gap Features)**: Longitudinal lab trends, patient timeline, AI diagnosis suggestions, genetics interpretation, drug-gene interaction checks, and clinical rules engine.
- **Data visualization**: Trend charts (Chart.js), clinical timeline, and alerts dashboard.

## Non-Goals (MVP)
- Real-time EHR integrations.
- Patient-facing portal or billing workflows.
- Clinical decision automation or prescribing.

## Data Inputs
- PDFs (lab reports, imaging summaries, clinical notes).
- Images/scans of lab sheets (OCR required).
- Structured notes (typed or pasted text).

## Data Outputs
- Structured extraction JSON (biomarkers, diagnoses, meds, procedures, genetics).
- Draft CHR with inline citations.
- Final CHR after clinician edits and sign-off.

## Tech Stack
- Backend: Python 3.11 + FastAPI.
- Frontend: Server-side Jinja2 templates with Chart.js for data visualization.
- OCR: Tesseract.
- PDF parsing: pdfplumber.
- LLM: OpenAI (pluggable via `llm_gateway.py`), with PHI policy enforcement.
- DB: PostgreSQL on Supabase.
- Vector search: pgvector extension (embeddings in Postgres).
- Storage: Supabase Storage (private bucket).

## Database (Free Options)
- Preferred: Postgres on Supabase (free tier).
- Alternative: Neon Postgres (free tier).
- Local dev: SQLite only for early prototyping.
- Extensions: pgvector enabled in Supabase for embeddings/RAG.

## Data Model (Draft)
- tenants(id, name, created_at)
- users(id, email, password_hash, role, tenant_id, created_at)
- patients(id, tenant_id, name, dob, created_at)
- documents(id, patient_id, filename, type, created_at)
- extractions(id, document_id, json_blob, created_at)
- embeddings(id, document_id, chunk_text, embedding_vector, created_at)
- chr_versions(id, patient_id, draft_json, status, created_at)
- edits(id, chr_version_id, field, old_value, new_value, editor, created_at)
- audit_logs(id, tenant_id, patient_id, actor, action, details, created_at)

## Workflow
1) Admin creates proper tenant and user accounts.
2) Clinician logs in (email/password).
3) Create patient and upload documents (scoped to tenant).
4) Extract text + tables (OCR as needed).
5) Parse into structured entities.
6) Normalize biomarker values and units.
7) Generate embeddings and store in pgvector (for RAG).
8) Generate CHR draft with citations.
9) Clinician edits and signs off.
10) Export and share finalized CHR.

## Workflow Visual

Workflow diagram (source of truth): [WORKFLOW.drawio](WORKFLOW.drawio).

Keep this file updated whenever the workflow changes.

## Evaluation and Quality
- Extraction accuracy (precision/recall on synthetic datasets).
- Citation correctness (claim supported by cited source).
- Time-to-report vs. manual baseline.
- Clinician edit rate (accept vs. revise).

## AI/ML Enhancements (Optional, Resume-Strengthening)
- Layout-aware document parsing.
- Section classifier.
- Hybrid extraction (rules + LLM).
- Schema-constrained JSON generation.
- Entity normalization to standards (LOINC/SNOMED).
- Retrieval-augmented drafting.
- Citation verifier.
- Consistency checks.
- Uncertainty tagging.
- Learning loop.
- Active learning labels.
- Style control templates.

## Privacy and Safety
- Use synthetic data for demos.
- Store PHI securely, encrypt at rest and in transit (future).
- Explicit clinician sign-off gate before sharing.
- No automated clinical decisions.
- **Strict Tenant Isolation**: Data from one clinic is invisible to others.

## Project Plan (Proposed)
Phase 1: Foundations (Week 1)
- Repo setup, DB schema, auth stub, file upload.
- Baseline parsing pipeline.

Phase 2: Extraction + Normalization (Week 2)
- Lab table parsing and biomarker normalization.
- Diagnoses/meds/procedures extraction.

Phase 3: CHR Drafting (Week 3)
- LLM prompt + citation formatting.
- Draft CHR JSON + HTML rendering.

Phase 4: Clinician Review & Security (Week 4)
- Edit UI, audit log, versioning, sign-off.
- **Authentication & RBAC Implementation.**
- Export to PDF/HTML.

## Risks and Mitigations
- OCR noise -> add confidence thresholds.
- Hallucinated statements -> force citation links.
- Data heterogeneity -> support common lab formats first.

## Open Questions
- Which DB provider -> Local Postgres/Supabase.
- LLM provider -> OpenAI/Local.
- UI -> Jinja Templates (MVP).
