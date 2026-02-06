# MedCHR Architecture (Current vs Target)

Date: 2026-02-06  
Scope: runtime architecture for enterprise healthcare-grade posture

## Current Implemented Architecture

```mermaid
flowchart LR
    Clinician["Clinician Browser (/ui)"] --> App["FastAPI App (API + UI)"]
    Service["Integration Service (API Key)"] --> App

    App --> Auth["AuthN/AuthZ Layer (RBAC, MFA, Step-Up, CSRF, Tenant IP Policy)"]
    Auth --> Session["Server-Side Sessions (ui_sessions)"]
    Auth --> Lockout["MFA Lockouts (mfa_lockouts)"]

    App --> DB["Postgres/Supabase (Tenant RLS, Audit Events, PHI Egress Events)"]
    App --> Storage["Supabase Storage (Private Bucket)"]
    App --> Presign["Signed Upload/Download URL Issuer"]
    Presign --> Storage

    App --> Worker["Background Worker (Redis Queue + DB Fallback)"]
    Worker --> OCR["OCR/Extraction Pipeline"]
    Worker --> LLM["LLM Gateway (Policy + Redaction + Egress Audit)"]
    LLM --> OpenAI["OpenAI API (Allowed PHI Processor)"]

    App --> Metrics["Metrics + Security Headers + Audit Chain"]
```

## Target Enterprise Healthcare Architecture

```mermaid
flowchart LR
    User["Clinician / Staff"] --> Edge["WAF + API Gateway + TLS Termination"]
    Partner["Partner System"] --> Edge
    Edge --> App["MedCHR Application Tier (FastAPI + Worker)"]

    IdP["Enterprise IdP (OIDC/SAML, Conditional Access)"] --> App
    App --> Policy["Central Policy Engine (Tenant, Role, Data Access, PHI Egress)"]
    App --> Secrets["Managed Secrets + KMS/HSM Keys"]

    App --> DB["Managed Postgres (RLS, PITR, Encrypted Backups, Read Replicas)"]
    App --> Obj["Object Storage (Private, Versioned, WORM Retention)"]
    App --> Queue["Durable Queue + Worker Autoscaling"]

    App --> OCR["Document/OCR Services"]
    App --> LLM["LLM Gateway (Prompt Guardrails, Redaction, DLP, Audit)"]
    LLM --> ApprovedLLM["Approved LLM Providers (BAA + Regional Controls)"]

    App --> Audit["Append-Only Audit + Hash Chain Verifier"]
    Audit --> SIEM["SIEM/SOC Pipeline (Alerting + Correlation)"]
    App --> Metrics["Metrics/Tracing/Logs (SLO + Incident Response)"]
```

## What Remains to Reach Target
- Move from app-level controls to defense-in-depth at edge/network (WAF, DDoS, mTLS/service mesh where required).
- Add managed key lifecycle (KMS/HSM rotation, break-glass process, key-use audit review).
- Expand audit evidence automation to include periodic access certification and control attestations.
- Add runtime detection controls (SIEM integration, anomaly detection on PHI access, on-call playbooks linked to alerts).
- Formalize deployment governance (environment segregation, change approval, mandatory production gate checks).

## LLM Position in Architecture
- The architecture is explicitly LLM-enabled.
- LLM usage is constrained behind `backend/app/llm_gateway.py` and treated as controlled PHI egress.
- In enterprise mode, only approved providers with BAAs should be enabled in `PHI_PROCESSORS`.
