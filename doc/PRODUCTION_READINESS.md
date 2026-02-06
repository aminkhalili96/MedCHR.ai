# Production Readiness Guide (HIPAA)

This document captures the production hardening steps added to MedCHR and the operational expectations for HIPAA-mode deployments.

## HIPAA Mode & PHI Processors
- `HIPAA_MODE=true` turns on stricter startup checks and secure session defaults.
- `PHI_PROCESSORS` is the allowlist of vendors that will receive PHI (e.g., `openai,supabase`).
- In HIPAA mode, the app refuses to start if required secrets or PHI processor allowlists are missing.
- Optional: `PHI_REDACTION_ENABLED=true` will redact common identifiers before sending text to external processors.

**What is an “approved PHI processor”?**  
It is a vendor that has signed a BAA (Business Associate Agreement) and is explicitly authorized to handle PHI for your use case.

## Enterprise Auth Settings
- SSO entrypoint: `/ui/sso/login`
- Optional provider envs:
  - OIDC: `OIDC_ENABLED`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, `OIDC_DISCOVERY_URL`
  - Azure AD: `AZURE_AD_ENABLED`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`
  - Google Workspace: `GOOGLE_WORKSPACE_ENABLED`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`
- Step-up MFA window for sensitive actions: `STEP_UP_WINDOW_MINUTES`
- Immutable retention sink directory: `RETENTION_IMMUTABLE_DIR` (required for execute-mode audit purges)
- Reverse-proxy trust controls for IP policy enforcement:
  - `TRUST_PROXY_HEADERS` (default `false`)
  - `TRUSTED_PROXY_IPS` (comma-separated proxy IPs/CIDRs that are allowed to provide `X-Forwarded-For` / `X-Real-IP`)
- Optional SSO policy envs:
  - `SSO_ALLOWED_DOMAINS` (comma-separated allowed email domains)
  - `SSO_ALLOWED_PROVIDERS` (comma-separated provider ids: `oidc`, `azure`, `google`)
  - `SSO_REQUIRE_VERIFIED_EMAIL` (default `true`)

## Security Controls Implemented
- **Authentication**: Email/Password login with bcrypt hashing.
- **MFA (TOTP)**: MFA setup secrets are stored server-side (short-lived `mfa_setup_tokens`), not in cookie sessions.
- **MFA secret encryption**: long-lived user TOTP secret is stored encrypted at rest (`users.mfa_secret_encrypted`) using app crypto key material (`MFA_SECRET_KEY` or derived from `APP_SECRET_KEY`).
- **MFA brute-force control**: account-scoped lockouts are persisted in `mfa_lockouts` and audited for lockout/clear events across login and step-up flows.
- **RBAC**: Role-based access (Admin vs Clinician) to enforce least privilege.
- **Multi-Tenancy**: Logical isolation of data using `tenant_id` (must be enforced consistently at every data access).
- API key auth for all non-UI endpoints (tenant-scoped env `API_KEYS` and/or DB-backed `api_keys` rows).
- **API key authorization**: per-key scopes (`read`/`write`/`admin`) and per-minute `rate_limit` are enforced.
- **Storage blast-radius reduction**: presigned upload/download URL flow for regular document access paths; service-role storage access retained only for privileged backend processing paths.
- CSRF protection for UI forms.
- Server-side UI sessions (`ui_sessions`) with signed opaque session-id cookies + strict referrer policy.
- CSP + security headers.
- Rate limiting via SlowAPI.
- Optional malware scanning with `clamdscan`.
- **PHI egress gate**: OpenAI calls flow through `backend/app/llm_gateway.py` wrappers (`create_chat_completion`, `create_embedding`) which enforce:
  - HIPAA processor allowlisting (`PHI_PROCESSORS`)
  - tenant-level AI policy (`tenant_phi_policies`)
  - egress auditing (`phi_egress_events`)
  - optional PHI redaction (`PHI_REDACTION_ENABLED`)
- **DB-level tenant isolation**: RLS policies enforced in `007_tenant_rls.sql` with request/worker tenant context.
- **RLS non-bypass mode**: `011_force_rls.sql` forces RLS on tenant-scoped tables.
- **Step-up MFA for sensitive actions**: in prod/HIPAA mode, share/finalize/report-adjacent admin actions require recent MFA verification.
- **Tenant network controls**: per-tenant IP allowlist is enforced for API-key and UI-authenticated traffic when configured.
- **Proxy trust boundary**: client IP extraction only trusts forwarded headers from explicitly trusted proxy IPs/CIDRs.
- **Web XSS hardening**: CSP `script-src` uses per-request nonce values and inline event handlers were removed from UI templates.
- **Audit chain integrity**: per-tenant concurrency-safe hash-chain state + verifier utility (`python -m backend.scripts.verify_audit_chain`).
- **Audit coverage guard**: PHI endpoint audit policy test (`backend/tests/test_audit_coverage.py`) fails on unaudited routes.
- **LLM gateway enforcement gate**: static validator ensures direct OpenAI imports are centralized in `backend/app/llm_gateway.py` (`python -m backend.scripts.validate_llm_gateway_usage`).
- **Runtime policy gate**: Docker/runtime policy checks for non-root user, healthcheck, and pinned dependencies (`python -m backend.scripts.validate_runtime_policy`).
- **Secret scan gate**: tracked files are scanned for common high-risk credential patterns (`python -m backend.scripts.scan_secrets`).
- **Evidence integrity**: evidence packs now include signed manifest + checksums (`evidence_manifest.json`).
- **Resilience drill evidence**: backup/restore drill script emits machine-readable reports (`python -m backend.scripts.backup_restore_drill`).

## Background Processing
Long-running tasks (OCR, embeddings, draft generation) can be queued when `JOB_QUEUE_ENABLED=true`.

Run the worker:
```bash
python -m backend.scripts.worker
```

API endpoints return `202` + `job_id` when queued. Use:
```
GET /jobs/{job_id}
```

## Assumed Scale (Reasonable Defaults)
These defaults align with an early production rollout:
- 5-25 concurrent clinicians.
- 1k-10k patients.
- 5-20 documents per patient (PDF + images).
- Typical upload size: 1-10MB per file, max 25MB configured.
- 100-500 embedding chunks per document.

## Migrations
All schema changes run via the migration runner:
```bash
python -m backend.scripts.init_db
```

## API Key Provisioning
Recommended for production/HIPAA mode: use DB-backed hashed keys.

Environment key format (supported, but less preferred than DB-backed keys):
- `API_KEYS="<tenant_uuid>:<key>,<tenant_uuid>:<key>"`
- In `HIPAA_MODE`/`APP_ENV=prod`, **unscoped** env keys (no tenant prefix) are rejected.

Create a key:
```bash
python -m backend.scripts.create_api_key --tenant-id <tenant_uuid> --name <service_name> --scopes read,write
```

Notes:
- Plaintext key is shown only once at creation time.
- Database stores only `api_keys.key_hash` (HMAC-SHA256 with `APP_SECRET_KEY`), never plaintext.
- `api_keys.scopes` and `api_keys.rate_limit` are actively enforced by request middleware.
- Startup validation in prod/HIPAA mode requires at least one configured key source (tenant-scoped env keys or active DB keys).

## Data Lifecycle
- Delete patient: `DELETE /patients/{patient_id}` (also removes storage objects)
- Delete document: `DELETE /documents/{document_id}`
- Purge old logs/jobs:
```bash
python -m backend.scripts.purge_data --execute
```
`purge_data` exports `audit_logs`, `audit_events`, and `phi_egress_events` to `RETENTION_EXPORT_DIR` (JSONL + `.sha256`) before deletion, writes immutable sink copies (`RETENTION_IMMUTABLE_DIR` / `--immutable-dir`), records `retention_manifests`, and only then proceeds with purge. It also purges expired/revoked `ui_sessions` and expired/used `mfa_setup_tokens`.

Generate evidence pack artifacts:
```bash
python -m backend.scripts.generate_evidence_pack
```

Validate cloud-agnostic healthcare controls:
```bash
python -m backend.scripts.validate_controls --require-db
```

Run backup/restore drill artifacts:
```bash
python -m backend.scripts.backup_restore_drill --output-dir data/drills/backup_restore --fail-on-errors
```

## Operational Notes
- Make sure TLS is enforced (HSTS is enabled in prod/HIPAA mode).
- Ensure your K8s ingress/controller passes correct host headers; if forwarded client-IP headers are used, set `TRUST_PROXY_HEADERS=true` and restrict `TRUSTED_PROXY_IPS` to ingress/LB addresses.
- Use secrets for all credentials and API keys.
- Run a Redis instance and set `REDIS_URL` for queue-backed jobs with low-latency worker claim loops.
- `/metrics` is available for Prometheus scraping; in prod/HIPAA mode it is admin-gated.
- CI generates SBOM (`sbom.cdx.json`) and build provenance attestation; enforce review of dependency-refresh PRs.
- CI enforces provider-neutral release gates: control validation, evidence generation, secret scan, runtime policy validation, LLM gateway validation, and container vulnerability scanning.
- Review and operationalize runbooks under `doc/runbooks/`:
  - `key_rotation.md`
  - `access_review.md`
  - `incident_response.md`
  - `breach_notification.md`
- Maintain provider-neutral baseline in `doc/CLOUD_AGNOSTIC_HEALTHCARE_BASELINE.md` until production cloud is finalized.

## Learning Notes (Why These Changes Exist)
- API keys + hashed passwords prevent accidental exposure of PHI through unauthenticated endpoints.
- CSRF and strict session cookies reduce web session abuse risk.
- Sanitized markdown blocks XSS from AI-generated content.
- Background jobs keep long tasks from blocking API workers.
- Migrations and indexes set the foundation for scale and safe schema evolution.
- Aggregated latest extractions avoid stale data while de-duplicating labs and diagnoses across documents.
- Embedding provenance (extraction_id + chunk offsets) ties citations back to the exact source text.
- LLM drafts are requested as JSON sections to keep interpretation and gaps consistent for review.
- Manual report edits override AI summaries without losing the original draft.

### New Learning Takeaways: Authentication & RBAC
- **Why Authentication?**: We moved from a single shared password to individual user accounts to ensure accountability (Audit Logs show exactly *who* did what) and security (revoking one user's access doesn't affect others).
- **Why RBAC?**: By separating "Admins" (who manage users) from "Clinicians" (who manage patients), we reduce the risk of accidental system-wide changes by daily users. This implementation follows the "Least Privilege" principle.
- **Why Multi-Tenancy?**: In healthcare SaaS, strictly separating data by "Tenant" (Clinic) is mandatory. We used "Row-Level Isolation" (adding `tenant_id` to every table) because it's cost-effective and simpler to manage than creating a separate database for every clinic.

## Engineering Pointers
- `codex.md` tracks the hardening plan and engineering conventions.
- `security_best_practices_report.md` tracks prioritized enterprise/HIPAA gaps found in the codebase.
