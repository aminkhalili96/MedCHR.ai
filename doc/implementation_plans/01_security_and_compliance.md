# Implementation Plan: Security & Compliance (Phase 1)

## Current Status (2026-02-06)
- Implemented in codebase:
  - TOTP MFA setup + challenge UI
  - Server-side UI sessions (`ui_sessions`)
  - SSO scaffolding (OIDC/Azure/Google + callback flow)
  - Step-up MFA for sensitive actions
  - RBAC permission helpers + enforcement on sensitive/admin routes
  - Append-only audit event write path (`audit_events`) + immutability trigger migration
- See also:
  - `codex.md` for phase-level completion status
  - `doc/PRODUCTION_READINESS.md` for operational expectations

## Objective
Establish a HIPAA-compliant security foundation focusing on Identity & Access Management (IAM), comprehensive audit logging, and data protection.

## 1. Authentication Upgrades (MFA & SSO)
**Gap**: No Multi-Factor Authentication (MFA) or Single Sign-On (SSO).

### 1.1 Multi-Factor Authentication (TOTP)
*   **Library**: `pyotp` + `qrcode`
*   **Schema Changes**:
    ```sql
    ALTER TABLE users ADD COLUMN mfa_secret TEXT;
    ALTER TABLE users ADD COLUMN mfa_enabled BOOLEAN DEFAULT FALSE;
    ALTER TABLE users ADD COLUMN backup_codes TEXT[]; -- JSON/Array of hashed codes
    ```
*   **Workflow**:
    1.  User goes to `/profile/security`.
    2.  Server generates secret `pyotp.random_base32()`.
    3.  Frontend displays QR code (`qrcode` gen).
    4.  User enters code to verify.
    5.  Server validates with `totp.verify(code)`.
    6.  Enforce MFA on login if `mfa_enabled` is true.

### 1.2 Enterprise SSO (SAML/OIDC)
*   **Strategy**: Use an abstraction layer (e.g., `python-social-auth` or a dedicated library like `pysaml2` / `authlib`). For MVP speed + enterprise value, recommend **Auth0** or **Descope** integration if budget allows, otherwise `Authlib` for OIDC with hospital IdPs.
*   **Implementation**:
    *   Add `sso_provider` and `sso_id` to `tenants` table.
    *   Implement `/auth/login/{tenant_slug}` to redirect to IdP.

## 2. Advanced Session Management
**Gap**: Basic cookies; no timeout or device tracking.

*   **Session Store**: Move from cookie-only to Redis-backed sessions (using `redis-py` + custom middleware) for immediate revocation.
*   **Idle Timeout**: Enforce 15-minute idle timeout in middleware.
*   **Schema**:
    ```sql
    CREATE TABLE user_sessions (
      id UUID PRIMARY KEY,
      user_id UUID REFERENCES users(id),
      ip_address INET,
      user_agent TEXT,
      last_activity TIMESTAMPTZ,
      expires_at TIMESTAMPTZ
    );
    ```

## 3. Comprehensive Audit Logging
**Gap**: Basic logging; mutable.

*   **Requirement**: Log *every* view of PHI (Read access).
*   **Implementation**:
    *   Create a `audit_events` table partitioned by date.
    *   **Fields**: `event_id`, `timestamp`, `actor_id`, `ip_address`, `resource_type` (Patient, document), `resource_id`, `action` (VIEW, EDIT, EXPORT), `outcome` (SUCCESS, DENIED), `metadata` (JSON).
    *   **Middleware**: Create an `AuditMiddleware` that attaches a context var.
    *   **Decorator**: `@audit_log(action="view_patient")` on API routes.
*   **immutability**: Use Postgres Write-Ahead Log (WAL) archiving or stream logs to AWS CloudWatch Logs / Datadog immediately.

## 4. IP Whitelisting
**Gap**: No network restrictions.

*   **Schema**: `ALTER TABLE tenants ADD COLUMN allowed_ips CIDR[];`
*   **Middleware**: Check `request.client.host` against tenant's `allowed_ips`. Reject 403 if not matched.

## 5. Security Headers & Crypto
*   **Encryption**: Ensure `pgcrypto` is used for sensitive fields if not using disk encryption.
*   **Headers**: rigorous CSP in `main.py`:
    ```python
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'nonce-...'; ..."
    ```

## Roadmap Tasks
- [x] Requirements: `pyotp`, `qrcode`, `authlib`, `redis` available in `backend/requirements.txt`.
- [x] DB Migrations: MFA columns + server-side sessions + audit events (`005_mfa_setup_tokens.sql`, `006_ui_sessions.sql`, `009_audit_events_immutability.sql`).
- [x] Backend: TOTP generation/verification + step-up endpoints.
- [x] Backend: Audit event middleware/helper path (centralized append-only write path).
- [x] Frontend: MFA setup + MFA challenge pages.
