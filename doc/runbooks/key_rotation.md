# Key Rotation Runbook

## Scope
- `APP_SECRET_KEY`
- DB-backed API keys (`api_keys`)
- External provider secrets (`OPENAI_API_KEY`, SSO client secrets, Supabase service role key)

## Cadence
- Quarterly scheduled rotation.
- Immediate rotation after suspected exposure.

## Procedure
1. Create replacement secret in secret manager.
2. Deploy app with dual-read period if supported.
3. For API keys:
   - create new key with `python -m backend.scripts.create_api_key ...`
   - migrate dependent services to new key
   - set `revoked_at` for old key
4. Verify:
   - auth success with new secret/key
   - revoked key returns `401`
5. Record change ticket, approver, and verification evidence.

## Evidence
- Audit events for key creation/revocation.
- Deployment ID and timestamp.
- Access review entry that confirms old key removal.
