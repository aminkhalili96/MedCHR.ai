# Access Review Runbook

## Scope
- UI users (`users` table, role assignments)
- API keys (`api_keys`)
- Break-glass/admin accounts

## Cadence
- Monthly operational review.
- Quarterly signed compliance review.

## Procedure
1. Export role snapshot:
   - `SELECT tenant_id, email, role FROM users ORDER BY tenant_id, role, email;`
2. Export active API keys:
   - `SELECT tenant_id, name, scopes, created_at, last_used_at FROM api_keys WHERE revoked_at IS NULL;`
3. Validate least privilege:
   - remove inactive users
   - downgrade over-privileged roles
   - revoke stale API keys
4. Verify MFA enabled for admin users.
5. Record approvals and remediation actions.

## Evidence
- Export artifacts.
- Ticket with reviewer + approver.
- Follow-up confirmation of revoked accounts/keys.
