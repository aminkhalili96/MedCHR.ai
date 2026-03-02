# Cloud-Agnostic Healthcare Baseline

Date: 2026-02-06  
Scope: Provider-neutral controls required before production PHI workloads

## Objective
Establish a healthcare-grade technical baseline that does not lock to any single cloud provider, while preserving a clear path to provider-specific implementation later.

## Control Domains
- **Identity & Access**: strong authentication, MFA/step-up, RBAC, tenant isolation, key lifecycle.
- **Data Protection**: encryption at rest/in transit, PHI egress controls, retention and purge safeguards.
- **Application Security**: secure headers/CSP, CSRF, rate limiting, upload validation, least-privilege API scopes.
- **Audit & Monitoring**: append-only audit logs, integrity verification, SIEM-compatible exports, incident alerting.
- **Resilience & Recovery**: backup strategy, restore drills, RTO/RPO evidence, retention manifest integrity.
- **Operational Governance**: runbooks, access reviews, key rotation cadence, evidence generation and control attestations.

## Provider-Neutral Build Outputs in This Repo
- Control validation script: `python -m backend.scripts.validate_controls`
- Compliance evidence pack: `python -m backend.scripts.generate_evidence_pack`
- Backup/restore drill report: `python -m backend.scripts.backup_restore_drill`
- Runtime policy gate: `python -m backend.scripts.validate_runtime_policy`
- LLM gateway usage gate: `python -m backend.scripts.validate_llm_gateway_usage`
- Secret scan gate: `python -m backend.scripts.scan_secrets`
- Operational runbooks: `doc/runbooks/*.md`
- Security/compliance architecture and roadmap:
  - `doc/ARCHITECTURE.md`
  - `doc/implementation_plans/07_enterprise_hipaa_execution_plan.md`

Execution note: control validation and LLM gateway validation resolve paths from the repository root, so command behavior is consistent when run from either project root or `backend/`.

## Current Control Validation Criteria
The provider-neutral validator currently verifies:
- runtime HIPAA posture (`APP_ENV`/`HIPAA_MODE`)
- secret hygiene (`APP_SECRET_KEY`, `MFA_SECRET_KEY`)
- proxy trust boundaries (`TRUST_PROXY_HEADERS`, `TRUSTED_PROXY_IPS`)
- PHI processor allowlist policy (`PHI_PROCESSORS`)
- immutable retention export target (`RETENTION_IMMUTABLE_DIR`)
- required runbook presence
- database connectivity + required compliance tables
- migration checksum coverage
- active tenant-scoped API key source presence (DB-backed keys and/or scoped env keys)
- audit hash-chain verification
- runtime policy compliance
- llm gateway centralization enforcement

## What Remains Provider-Specific (Later)
- WAF/L7 protections and DDoS policy implementation
- managed KMS/HSM key hierarchy and rotation policies
- SIEM connector and SOC alert routing
- backup/PITR provider controls and restore orchestration
- production network segmentation and private service connectivity

## Recommended Execution Flow (No Cloud Lock Yet)
1. Keep app/runtime controls cloud-agnostic and test-gated.
2. Use control validation + evidence packs as release gates.
3. Select provider before production PHI launch, then map each control to provider-native services.
4. Preserve the same control IDs/evidence shape so audit history remains continuous through provider adoption.
5. Keep evidence packs and drill reports on a rolling cadence (`controls-evidence.yml`, monthly) with external long-term retention.
