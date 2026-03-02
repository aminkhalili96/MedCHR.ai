# Implementation Plan: Enterprise Scale (Phase 5)

## Objective
Support large hospital systems with multi-hierarchy tenancy, custom branding, and usage analytics.

## 1. Hierarchical Multi-Tenancy
**Gap**: Flat "Tenant" structure doesn't support "Hospital System -> Region -> Clinic".

*   **Schema Update**:
    ```sql
    ALTER TABLE tenants ADD COLUMN parent_id UUID REFERENCES tenants(id);
    ALTER TABLE tenants ADD COLUMN type TEXT; -- 'organization', 'facility', 'department'
    ```
*   **Data Isolation**: Ensure data access cascades (Org Admin sees all; Clinic Admin sees only Clinic).

## 2. White-Labeling & Customization
**Gap**: Single generic UI.

*   **Tenant Config**:
    ```sql
    ALTER TABLE tenants ADD COLUMN branding_config JSONB;
    -- { "primary_color": "#0055AA", "logo_url": "s3://...", "custom_domain": "portal.clinic.com" }
    ```
*   **Frontend**: React context `ThemeContext` loads these values on login and applies CSS variables dynamically.

## 3. Analytics & Reporting
**Gap**: No admin insights.

*   **Tech Stack**: ClickHouse or Postgres read-replica for OLAP queries.
*   **Metrics**:
    *   Time saved per report (Edit time vs generic baseline).
    *   Document volume trends.
    *   User activity heatmaps.
*   **Dashboard**: Recharts/Victory charts in the Admin portal.

## 4. API Tiers & Rate Limiting
**Gap**: One size fits all.

*   **Implementation**:
    *   tier column in `tenants`.
    *   Use `slowapi` or API Gateway (Kong/Zuul) to enforce limits based on tier.
    *   *Free*: 100 req/day; *Enterprise*: 10,000 req/day.

## Roadmap Tasks
- [ ] DB Migration: Hierarchical tenants columns.
- [ ] DB Migration: Branding config JSONB.
- [ ] Backend: Middleware to inject tenant branding into responses.
- [ ] Frontend: Dynamic stylesheet injection.
- [ ] Backend: Reporting API endpoints (aggregates).
