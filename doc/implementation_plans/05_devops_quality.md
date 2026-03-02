# Implementation Plan: DevOps & Quality (Phase 2 & Foundation)

## Current Status (2026-02-06)
- Implemented in codebase:
  - CI workflow with tests, migration smoke run, Bandit scan, and dependency audit
  - Hardened multi-stage Dockerfile with non-root runtime, no baked `.env`, and healthcheck
  - Prometheus-style `/metrics` endpoint with request latency instrumentation
  - Retention export-before-purge workflow for audit data (`purge_data.py`)
- Still pending:
  - Terraform/IaC module implementation
  - Full integration-test and E2E test suites
  - Full centralized structured logging pipeline deployment (ELK/Datadog sink)

## Objective
Achieve operational excellence, disaster recovery, and high availability.

## 1. CI/CD Pipeline (GitHub Actions)
**Gap**: No automated testing or deployment.

*   **Workflows**:
    *   `ci.yml`: Runs on PR. `install deps`, `lint (ruff/mypy)`, `test (pytest)`, `security-scan (bandit)`.
    *   `cd.yml`: Runs on merge to main. `build docker`, `push ECR/GCR`, `deploy helm`.
*   **Quality Gates**: Block merge if coverage < 80%.

## 2. Infrastructure as Code (Terraform)
**Gap**: Manual setup.

*   **Modules**:
    *   `vpc`: Networking (private subnets for DB).
    *   `rds`: Postgres + pgvector.
    *   `k8s`: EKS/GKE cluster.
    *   `buckets`: S3 private buckets with lifecycle policies.
*   **State Management**: Remote state in S3 + DynamoDB locking.

## 3. Observability & Monitoring
**Gap**: Local logging only.

*   **Logs**: Strucutred JSON logging. Ship to Datadog/ELK via FluentBit.
*   **Metrics**: Prometheus endpoint (`/metrics`) exposing request latency, DB connection pool status, job queue depth.
*   **Tracing**: OpenTelemetry (OTEL) instrumentation for FastAPI. View traces in Jaeger/Honeycomb to see "API -> DB -> OpenAI" latency waterfall.

## 4. Testing Strategy
**Gap**: Minimal tests.

*   **Unit Tests**: Mock DB and OpenAI. Test logic of parsers.
*   **Integration Tests**: Spin up Docker Compose (test DB). Hit actual API endpoints.
*   **End-to-End (E2E)**: Playwright tests. Login -> Upload PDF -> Verify Extraction -> Logout.
*   **Load Testing**: k6 script to simulate 50 concurrent clinicians generating reports.

## Roadmap Tasks
- [x] Repo: Add `.github/workflows/ci.yml`.
- [x] Repo: Dockerize app (`Dockerfile` multi-stage).
- [ ] Infra: Create `terraform/` directory.
- [x] Code: Add metrics endpoint/instrumentation (`/metrics` + request middleware).
- [ ] Code: Complete `structlog` end-to-end wiring for production sink.
- [ ] Tests: Write `tests/integration/` suite.
