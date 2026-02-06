# MedCHR MVP

Clinician-in-the-loop pipeline for turning raw patient records into editable Client Health Reports (CHRs).

## Quick Start
Run all commands from the project root (one level up from this file).

1) Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2) Copy env template:

```bash
cp .env.example .env
```

Fill in:
- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `APP_SECRET_KEY`
- `APP_BASE_URL` (for SSO callback URLs)
- `API_KEYS` (optional if you provision DB-backed keys; in `HIPAA_MODE`/`APP_ENV=prod` this must be tenant-scoped: `<tenant_uuid>:<key>`)
- `HIPAA_MODE` (set `true` for production)
- `PHI_PROCESSORS` (e.g., `openai,supabase` when BAAs are in place)
- `MFA_SECRET_KEY` (optional Fernet key; if omitted, derived from `APP_SECRET_KEY`)
- `REDIS_URL` (optional, enables Redis-backed job queue)
- `RETENTION_IMMUTABLE_DIR` (required in execute mode when purging audit/PHI data)
- `TRUST_PROXY_HEADERS` and `TRUSTED_PROXY_IPS` (required when running behind ingress/LB and enforcing tenant IP allowlists)

Optional enterprise identity settings:
- `OIDC_ENABLED`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, `OIDC_DISCOVERY_URL`
- `AZURE_AD_ENABLED`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`
- `GOOGLE_WORKSPACE_ENABLED`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`
- `SSO_ALLOWED_DOMAINS`, `SSO_ALLOWED_PROVIDERS`, `SSO_REQUIRE_VERIFIED_EMAIL`

3) Create Supabase Storage bucket:
- **Storage → New bucket →** `medchr-uploads` (private)

4) Enable pgvector in Supabase:
- **Database → Extensions →** `vector`

5) Apply DB migrations:

```bash
python -m backend.scripts.init_db
```

6) Run API:

```bash
uvicorn app.main:app --app-dir backend --reload
```

7) One-command dev run (applies schema + starts API):

```bash
./run_dev.sh
```

8) Open UI:
- Visit `http://127.0.0.1:8000/ui`
- Create a tenant + first admin user (once):
  - `python -m backend.scripts.bootstrap_admin`
- Login with the admin email/password you bootstrapped
- Embeddings debug: `http://127.0.0.1:8000/ui/embeddings`
- RAG viewer: open a patient and click “View RAG Top-K Chunks”

9) Provision an API key (recommended for production/HIPAA mode):

```bash
python -m backend.scripts.create_api_key \
  --tenant-id 00000000-0000-0000-0000-000000000000 \
  --name integration-service \
  --scopes read,write
```

Notes:
- The plaintext key is shown once.
- The database stores only an HMAC hash (`api_keys.key_hash`).
- API authentication accepts either tenant-scoped `API_KEYS` from env (format: `<tenant_uuid>:<key>`) or active DB-backed keys.
- API-key scopes are enforced (`read` / `write` / `admin`) and DB-backed per-key `rate_limit` is enforced per-minute.
- UI sessions are server-side (`ui_sessions` table); cookies carry only a signed opaque session id.
- Sensitive UI actions require step-up MFA in `HIPAA_MODE`/`APP_ENV=prod`.
- MFA secrets are stored encrypted at rest (`users.mfa_secret_encrypted`).
- MFA lockouts are account-scoped and persisted in `mfa_lockouts` (survive browser/session resets).
- SSO login entrypoint is `/ui/sso/login` when any SSO provider is enabled.
- Tenant IP allowlists are enforced for both API-key and UI-authenticated requests when configured.
- Forwarded client IP headers are only trusted when `TRUST_PROXY_HEADERS=true` and the socket source IP is in `TRUSTED_PROXY_IPS`.

## Background Jobs (Recommended)
Enable async processing (OCR, embeddings, CHR draft) by setting `JOB_QUEUE_ENABLED=true` and running a worker:

```bash
python -m backend.scripts.worker
```

If `REDIS_URL` is configured, enqueue/claim runs through Redis (`REDIS_QUEUE_NAME`, default `medchr:jobs`) with DB status tracking as fallback.

## Mock Data Import (Bulk)
To load the synthetic dataset into Postgres and Supabase Storage:

```bash
python -m backend.scripts.import_mock_data
```

Optional flags:
- `--patient patient_a` (repeat for multiple)
- `--skip-embed` (skip embeddings)
- `--skip-draft` (skip CHR draft)

## Synthetic Data Generation (Scale Testing)
Generate 100 realistic, complex hospital-grade patient records directly into the database (no local files created):

```bash
cd MedCHR.ai
python -m backend.scripts.generate_scale_data
```

This script uses "Clinical Archetypes" (e.g., Heart Failure, Diabetes, COPD) to create medically consistent patient histories with matching diagnoses, medications, and lab results. Embeddings are generated via the OpenAI API.

To verify the generated data:

```bash
python -m backend.scripts.verify_scale_data
```

## Data Lifecycle
- Delete patient: `DELETE /patients/{patient_id}`
- Delete document: `DELETE /documents/{document_id}`
- Purge old audit/job records:
```bash
python -m backend.scripts.purge_data --execute
```
`purge_data` exports retention data (`audit_logs`, `audit_events`, `phi_egress_events`) to `RETENTION_EXPORT_DIR` with SHA-256 checksum files before deletion.
In execute mode, immutable export confirmation is enforced via `RETENTION_IMMUTABLE_DIR` (or `--immutable-dir`) and manifest rows in `retention_manifests`.

Generate a compliance evidence pack:
```bash
python -m backend.scripts.generate_evidence_pack
```

Validate cloud-agnostic healthcare controls:
```bash
python -m backend.scripts.validate_controls --require-db
```

Run backup/restore drill and produce machine-readable artifacts:
```bash
python -m backend.scripts.backup_restore_drill --output-dir data/drills/backup_restore --fail-on-errors
```

## Key Endpoints
- `GET /health` — liveness
- `GET /ready` — readiness (DB + storage)
- `GET /metrics` — Prometheus metrics (admin-only in prod/HIPAA mode)
- `POST /patients` — create patient
- `DELETE /patients/{patient_id}` — delete patient + documents
- `POST /patients/{patient_id}/documents` — upload document
- `POST /patients/{patient_id}/documents/presign-upload` — issue signed upload URL/token
- `POST /patients/{patient_id}/documents/register-upload` — register presigned upload as a document
- `DELETE /documents/{document_id}` — delete document + storage
- `GET /documents/{document_id}/download-url` — issue signed download URL
- `POST /documents/{document_id}/extract` — extract raw text + structured data
- `POST /documents/{document_id}/embed` — store embeddings in pgvector
- `POST /chr/draft` — generate CHR draft
- `GET /jobs/{job_id}` — check async job status

## OCR Notes
This MVP uses Tesseract. Install it (macOS):

```bash
brew install tesseract
```

## Project Docs
- `codex.md` — engineering + compliance notes (living doc)
- `security_best_practices_report.md` — prioritized HIPAA/enterprise gaps + roadmap
- `doc/PROJECT_DOC.md` — scope, plan, interview prep
- `doc/ARCHITECTURE.md` — current and target enterprise healthcare architecture diagrams
- `doc/CLOUD_AGNOSTIC_HEALTHCARE_BASELINE.md` — provider-neutral healthcare control baseline
- `WORKFLOW.drawio` — workflow diagram source of truth
- `doc/PRODUCTION_READINESS.md` — HIPAA hardening checklist + operational notes
- `doc/K8S_DEPLOYMENT.md` — Kubernetes deployment examples
- `doc/runbooks/*.md` — key rotation, access reviews, incident and breach response

## Production Security Defaults
- OpenAPI docs are disabled automatically when `APP_ENV=prod` or `HIPAA_MODE=true`.
- Outbound model calls flow through `backend/app/llm_gateway.py`, enforcing `PHI_PROCESSORS`, optional tenant PHI policy (`tenant_phi_policies`), and `phi_egress_events` audit records.
- CSP is nonce-based for scripts (no `unsafe-inline` in `script-src`) and UI templates avoid inline event handlers.
