# Enterprise HIPAA Execution Plan (Build Plan)

Date: 2026-02-06  
Scope: MedCHR.ai application, infrastructure, and operational controls

## Goal
Move MedCHR from current hardened MVP posture to enterprise healthcare-grade readiness with HIPAA-aligned technical safeguards and auditability.

## Execution Status (2026-02-06)
- [x] E0.1 API key scope + rate-limit enforcement
- [x] E0.2 MFA anti-bruteforce controls
- [x] E0.3 FORCE RLS
- [x] E0.4 SSO trust hardening
- [x] E0.5 Tenant IP allowlisting activation
- [x] E0.6 Encrypt TOTP secrets at rest
- [x] E0.7 Storage service-role blast-radius reduction via presigned flow
- [x] E0.8 Reverse-proxy trust boundaries for tenant IP controls
- [x] E1.1 Concurrency-safe audit hash chain + verifier
- [x] E1.2 Immutable retention sink + manifests
- [x] E1.3 PHI audit coverage map test gate
- [x] E1.4 Nonce-based CSP for UI script execution
- [x] E2.1 Reproducible dependency management
- [x] E2.2 SBOM + signed artifact attestation + vulnerability gate
- [x] E2.3 Migration integrity checksum controls
- [x] E3.1 Operational compliance runbooks
- [x] E3.2 Evidence automation + scheduled report job
- [x] E3.3 Cloud-agnostic control validation baseline
- [x] E3.4 Release-blocking technical healthcare gates

## Delivery Model
- Run as 4 execution waves with strict acceptance gates.
- Do not start Wave N+1 until Wave N exit criteria are met.
- Every control must ship with:
  - code changes,
  - migration(s) if needed,
  - tests,
  - runbook/update in docs.

---

## Wave 0 (P0): Close High-Risk Gaps (2-4 weeks)

### E0.1 API key scope + rate-limit enforcement
**Why:** Keys are tenant-scoped but scope/rate limit are not enforced.

**Build tasks**
- Add per-route scope requirement helper and enforce on all API endpoints.
- Enforce `api_keys.rate_limit` with request counters/windowed limiter.
- Add key revocation and “disabled key” behavior test coverage.

**Code targets**
- `backend/app/security.py`
- `backend/app/main.py`
- `backend/app/clinical.py`
- `backend/app/gap_features.py`

**Acceptance criteria**
- Requests without required scope return `403`.
- Revoked keys always return `401`.
- Rate-limit breaches return `429` with audit event.

---

### E0.2 MFA anti-bruteforce controls
**Why:** MFA endpoints are not currently rate-limited/lockout protected.

**Build tasks**
- Add limiter to `/ui/mfa-verify` and `/ui/step-up`.
- Add short lockout after repeated failures (session + user dimension).
- Emit audit events for lockout and unlock.

**Code targets**
- `backend/app/main.py`
- `frontend/templates/mfa_challenge.html`

**Acceptance criteria**
- Repeated bad codes trigger lockout and block verification attempts.
- Lockout events visible in audit trail.

---

### E0.3 Enforce non-bypass tenant isolation at DB role level
**Why:** RLS enabled but not forced.

**Build tasks**
- Add migration to `FORCE ROW LEVEL SECURITY` on tenant tables.
- Ensure app DB role cannot bypass RLS.
- Add integration test validating cross-tenant denial through DB policy.

**Code targets**
- `backend/sql/migrations/010_force_rls.sql` (new)
- `backend/tests/` integration test

**Acceptance criteria**
- Cross-tenant reads/writes are denied even under accidental query mistakes.

---

### E0.4 SSO trust hardening
**Why:** SSO provisioning needs stricter identity assertions.

**Build tasks**
- Require `email_verified=true` for OIDC providers.
- Enforce tenant domain allowlist and optional provider allowlist.
- Log denied SSO provisioning attempts.

**Code targets**
- `backend/app/sso.py`
- `backend/app/main.py`

**Acceptance criteria**
- Unverified email or domain mismatch cannot create/login users.

---

### E0.5 Activate tenant IP allowlisting
**Why:** IP allowlist module exists but is not wired.

**Build tasks**
- Apply IP allowlist check after auth context is established.
- Respect reverse-proxy header handling safely.
- Add admin UX/docs for managing allowlist values.

**Code targets**
- `backend/app/ip_whitelist.py`
- `backend/app/main.py`
- `doc/PRODUCTION_READINESS.md`

**Acceptance criteria**
- Blocked IPs receive `403`; allowed IPs proceed.
- Policy is tenant-specific and auditable.

---

### E0.6 Encrypt TOTP secrets at rest
**Why:** `users.mfa_secret` is currently plaintext.

**Build tasks**
- Introduce envelope encryption/decryption wrapper (KMS-backed or secret-key fallback for non-prod).
- Migrate `mfa_secret` to encrypted value column.
- Rotate and re-enroll flow for existing users.

**Code targets**
- `backend/app/auth.py`
- `backend/app/main.py`
- `backend/sql/migrations/011_encrypt_mfa_secret.sql` (new)

**Acceptance criteria**
- No plaintext TOTP secret remains in DB.
- MFA still works across login + step-up flows.

---

### E0.7 Reduce storage service-role blast radius
**Why:** all storage ops use service-role key.

**Build tasks**
- Move upload/download to presigned URL flow where possible.
- Keep service-role operations only for privileged maintenance paths.
- Add strict path/tenant ownership assertions before issuing URLs.

**Code targets**
- `backend/app/storage.py`
- `backend/app/main.py`

**Acceptance criteria**
- Regular user document access does not require direct service-role use in request path.

---

### E0.8 Enforce reverse-proxy trust boundaries for IP controls
**Why:** trusting forwarded client-IP headers from untrusted sources can bypass tenant IP allowlist policy.

**Build tasks**
- Add explicit proxy trust settings (`TRUST_PROXY_HEADERS`, `TRUSTED_PROXY_IPS`).
- Only honor `X-Forwarded-For` / `X-Real-IP` when socket source is trusted proxy IP/CIDR.
- Ensure legacy middleware paths never blindly trust forwarding headers.

**Code targets**
- `backend/app/config.py`
- `backend/app/ip_whitelist.py`
- `doc/PRODUCTION_READINESS.md`

**Acceptance criteria**
- Untrusted callers cannot spoof client IP via forwarding headers.
- Tenant IP allowlists use trusted client IP extraction in API/UI auth contexts.

---

## Wave 1 (P1): Audit, Integrity, and Retention Hardening (2-3 weeks)

### E1.1 Concurrency-safe audit hash chain
**Why:** hash-chain lookup can fork under concurrent inserts.

**Build tasks**
- Add per-tenant chain state table or locking strategy.
- Compute event hash using deterministic previous hash from locked state.
- Add verifier utility and CI integrity check.

**Code targets**
- `backend/sql/migrations/012_audit_chain_locking.sql` (new)
- `backend/app/security_audit.py`
- `backend/scripts/` verifier script

**Acceptance criteria**
- Under concurrent insert load, chain remains linear and verifiable.

---

### E1.2 External immutable retention sink (WORM)
**Why:** local export is not sufficient for enterprise compliance evidence.

**Build tasks**
- Extend purge export to immutable object storage target (S3 Object Lock / equivalent).
- Persist object URI + checksum + export manifest in DB.
- Document retention/legal hold runbook.

**Code targets**
- `backend/scripts/purge_data.py`
- `backend/sql/migrations/013_retention_manifests.sql` (new)
- `doc/PRODUCTION_READINESS.md`

**Acceptance criteria**
- Purge proceeds only after successful immutable export confirmation.

---

### E1.3 Full audit coverage map
**Why:** ensure every PHI read/write/export path is covered.

**Build tasks**
- Enumerate all PHI-touching routes/jobs.
- Add missing audit events for any uncovered path.
- Add automated coverage test that fails on new unaudited PHI endpoints.

**Code targets**
- `backend/app/main.py`
- `backend/app/clinical.py`
- `backend/app/gap_features.py`
- tests under `backend/tests/`

**Acceptance criteria**
- 100% PHI endpoint coverage in audit policy tests.

---

### E1.4 Strict CSP script policy for UI
**Why:** reducing inline script execution paths lowers XSS blast radius for PHI-facing UIs.

**Build tasks**
- Add nonce-based CSP `script-src` in middleware.
- Remove inline event handlers from templates and use nonce-scoped scripts.
- Add tests ensuring CSP scripts rely on nonce and not `unsafe-inline`.

**Code targets**
- `backend/app/main.py`
- `frontend/templates/*.html`
- `backend/tests/test_security_headers.py`

**Acceptance criteria**
- `script-src` contains nonce values and excludes `unsafe-inline`.
- UI behavior remains functional without inline event handlers.

---

## Wave 2 (P2): DevSecOps and Supply Chain (2-3 weeks)

### E2.1 Reproducible dependency management
**Build tasks**
- Pin all Python dependencies (lock file strategy).
- Add weekly dependency refresh workflow with review gate.

**Code targets**
- `backend/requirements.txt` (or generated lock file)
- CI workflows

**Acceptance criteria**
- Deterministic builds across environments.

---

### E2.2 SBOM + signed artifacts + vulnerability gate
**Build tasks**
- Generate SBOM in CI.
- Add artifact signing/attestation.
- Fail build on high/critical vulnerabilities unless explicitly waived.

**Code targets**
- `.github/workflows/ci.yml`
- release/build workflows

**Acceptance criteria**
- Every build has SBOM + signature and enforceable vuln policy.

---

### E2.3 Migration integrity controls
**Build tasks**
- Extend `schema_migrations` with checksum/signature.
- Validate checksum before applying migration.
- Fail fast on drift/tampering.

**Code targets**
- `backend/scripts/migrate.py`
- new migration for metadata columns

**Acceptance criteria**
- Modified historical migration file is detected and blocked.

---

## Wave 3 (P3): Enterprise Operations and HIPAA Program Readiness (ongoing)

### E3.1 Operational compliance controls
**Build tasks**
- Complete key rotation runbooks and quarterly cadence.
- Access review process (RBAC + API keys + break-glass).
- Incident response and breach notification runbooks.

**Artifacts**
- `doc/runbooks/*.md`
- policy docs linked from `doc/PRODUCTION_READINESS.md`

---

### E3.2 Evidence automation
**Build tasks**
- Create evidence export pack: audit integrity report, retention manifests, access review logs, vulnerability reports.
- Schedule periodic controls report generation.

**Acceptance criteria**
- One-command evidence package available for audit windows.

---

### E3.3 Cloud-agnostic control validation baseline
**Why:** maintain healthcare-grade readiness before cloud-provider lock-in.

**Build tasks**
- Implement provider-neutral control checks for runtime, database, and operations controls.
- Include control validation output in compliance evidence artifacts.
- Document baseline controls and transition path to provider-native implementation.

**Code targets**
- `backend/scripts/validate_controls.py`
- `backend/scripts/generate_evidence_pack.py`
- `doc/CLOUD_AGNOSTIC_HEALTHCARE_BASELINE.md`

**Acceptance criteria**
- Control validator reports pass/fail/warn per control ID.
- Evidence pack contains control validation summary for audit review.

---

### E3.4 Release-blocking technical healthcare gates
**Why:** prevent drift from healthcare-grade controls between releases.

**Build tasks**
- Enforce control validation as mandatory CI gate (`--require-db --fail-on-failures`).
- Add mandatory runtime policy, secret scan, LLM gateway usage, and container vulnerability gates.
- Require backup/restore drill artifacts and signed evidence manifests in controls workflows.

**Code targets**
- `.github/workflows/ci.yml`
- `.github/workflows/controls-evidence.yml`
- `backend/scripts/scan_secrets.py`
- `backend/scripts/validate_runtime_policy.py`
- `backend/scripts/validate_llm_gateway_usage.py`
- `backend/scripts/backup_restore_drill.py`

**Acceptance criteria**
- CI blocks release when any gate fails.
- Monthly controls workflow emits control report, drill report, evidence report, and signed manifest.

---

## Cross-Cutting Quality Gates (apply to every epic)
- Unit tests and integration tests added for each control.
- Security regression tests included for authz/tenant isolation.
- Documentation updated in:
  - `README.md`
  - `doc/PRODUCTION_READINESS.md`
  - `codex.md`
- Backward-compatible migrations with rollback notes.

---

## Recommended Implementation Order (Strict)
1. E0.1 API scope/rate-limit enforcement  
2. E0.2 MFA anti-bruteforce  
3. E0.3 FORCE RLS  
4. E0.6 Encrypt TOTP secrets  
5. E0.4 SSO trust hardening  
6. E0.5 IP allowlist activation  
7. E0.7 Storage blast-radius reduction  
8. E1.1 Audit chain safety  
9. E1.2 Immutable retention sink  
10. E1.3 Audit coverage map  
11. E2.1/E2.2/E2.3 DevSecOps hardening  
12. E3 operational controls and evidence automation

---

## Exit Criteria (Enterprise-Hardening Build Complete)
- Tenant isolation enforced at app + DB levels with bypass protections.
- Strong authn/authz with step-up and enforced scoped API access.
- PHI egress policy enforced globally with immutable evidence trail.
- Audit trail append-only, integrity-verifiable, and externally retained.
- Build/release pipeline meets reproducibility and security gating requirements.
- Operational HIPAA controls documented and exercised.
