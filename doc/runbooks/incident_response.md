# Incident Response Runbook

## Trigger
Suspected compromise, PHI exposure, unauthorized access, integrity failure, or service disruption with security impact.

## Response Steps
1. **Triage (0-30 min)**
   - classify severity
   - open incident channel and incident ticket
   - assign incident commander
2. **Containment (30-120 min)**
   - revoke affected API keys/sessions
   - isolate impacted services
   - block malicious source IPs (tenant/global controls)
3. **Investigation**
   - collect audit events and retention manifests
   - verify audit hash chain (`python -m backend.scripts.verify_audit_chain`)
   - preserve forensic artifacts
4. **Eradication + Recovery**
   - remove root cause
   - rotate secrets
   - restore service with heightened monitoring
5. **Post-incident**
   - postmortem within 5 business days
   - CAPA actions with owners/dates

## Required Artifacts
- Timeline of actions
- Affected systems/data classes
- Audit log exports and integrity report
- Stakeholder notification records
