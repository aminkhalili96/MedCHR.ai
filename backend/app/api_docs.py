"""
API Documentation Generator

Generates OpenAPI/Swagger documentation for the MedCHR API.

Gap Reference: F05
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi_schema(app: FastAPI) -> dict:
    """
    Generate custom OpenAPI schema with enhanced documentation.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="MedCHR.ai API",
        version="2.0.0",
        description="""
# MedCHR.ai Clinical Health Report API

Hospital-grade API for clinical document processing and CHR generation.

## Authentication

All API endpoints require authentication via API key:
- Pass `X-API-Key` header with your API key
- Or use Bearer token authentication for session-based access

## Rate Limits

- Default: 1000 requests/hour
- Extraction endpoints: 100 requests/hour
- Contact support for higher limits

## Data Formats

### Extraction Response
All extraction endpoints return structured clinical data:
- Labs with LOINC codes
- Medications with RxNorm codes
- Diagnoses with SNOMED codes
- Safety alerts for critical values

### FHIR Support
Export endpoints support FHIR R4 format for interoperability.

## Webhooks

Configure webhooks for:
- Extraction complete
- Critical value detected
- Document signed
        """,
        routes=app.routes,
        tags=[
            {"name": "patients", "description": "Patient management"},
            {"name": "documents", "description": "Document upload and processing"},
            {"name": "extractions", "description": "Structured data extraction"},
            {"name": "chr", "description": "Clinical Health Report generation"},
            {"name": "clinical", "description": "Clinical data management"},
            {"name": "FHIR", "description": "FHIR R4 interoperability"},
            {"name": "admin", "description": "Administrative operations"},
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token from login"
        }
    }
    
    # Add common response schemas
    openapi_schema["components"]["schemas"]["Error"] = {
        "type": "object",
        "properties": {
            "detail": {"type": "string"},
            "code": {"type": "string"},
            "path": {"type": "string"}
        }
    }
    
    openapi_schema["components"]["schemas"]["LabResult"] = {
        "type": "object",
        "properties": {
            "test_name": {"type": "string", "example": "Glucose"},
            "value": {"type": "string", "example": "126"},
            "unit": {"type": "string", "example": "mg/dL"},
            "flag": {"type": "string", "enum": ["H", "L", "N", "C"]},
            "reference_range": {"type": "string", "example": "70-99"},
            "confidence": {"type": "number", "example": 0.95},
            "loinc": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "name": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            }
        }
    }
    
    openapi_schema["components"]["schemas"]["Medication"] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "example": "Metformin"},
            "dosage": {"type": "string", "example": "500mg"},
            "frequency": {"type": "string", "example": "BID"},
            "route": {"type": "string", "example": "PO"},
            "status": {"type": "string", "enum": ["active", "discontinued", "prn"]},
            "rxnorm": {
                "type": "object",
                "properties": {
                    "rxcui": {"type": "string"},
                    "name": {"type": "string"}
                }
            }
        }
    }
    
    openapi_schema["components"]["schemas"]["Diagnosis"] = {
        "type": "object",
        "properties": {
            "condition": {"type": "string", "example": "Type 2 Diabetes"},
            "status": {"type": "string", "enum": ["active", "resolved", "negated", "historical"]},
            "code": {"type": "string", "example": "E11.9"},
            "confidence": {"type": "number", "example": 0.9},
            "snomed": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "display": {"type": "string"}
                }
            }
        }
    }
    
    openapi_schema["components"]["schemas"]["SafetyAlert"] = {
        "type": "object",
        "properties": {
            "test": {"type": "string"},
            "value": {"type": "number"},
            "severity": {"type": "string", "enum": ["CRITICAL", "HIGH", "MODERATE"]},
            "direction": {"type": "string"},
            "action": {"type": "string"}
        }
    }
    
    openapi_schema["components"]["schemas"]["CHRDraft"] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sections": {
                "type": "object",
                "properties": {
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "interpretation": {"type": "string"},
                    "recommendations": {"type": "array", "items": {"type": "string"}}
                }
            },
            "citations": {"type": "array", "items": {"type": "object"}},
            "audit_report": {
                "type": "object",
                "properties": {
                    "is_verified": {"type": "boolean"},
                    "issues": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Example endpoint documentation
ENDPOINT_DOCS = {
    "upload_document": {
        "summary": "Upload a medical document",
        "description": """
Upload a PDF or image document for processing.

Supported formats:
- PDF (application/pdf)
- PNG (image/png)
- JPEG (image/jpeg)

The document will be:
1. Classified by type (lab, consult, imaging, etc.)
2. OCR processed if needed
3. Queued for extraction
        """,
        "responses": {
            200: {"description": "Document uploaded and queued for processing"},
            400: {"description": "Invalid file format"},
            413: {"description": "File too large (max 50MB)"}
        }
    },
    "generate_chr": {
        "summary": "Generate Clinical Health Report",
        "description": """
Generate a comprehensive CHR from patient data.

Features:
- Retrieves relevant context using hybrid RAG
- Generates clinician-facing summary
- Cites sources with page references
- Runs AI auditor to verify claims
- Detects critical values and drug interactions
        """,
        "responses": {
            200: {"description": "CHR generated successfully"},
            404: {"description": "Patient not found"},
            422: {"description": "Insufficient data for report"}
        }
    }
}
