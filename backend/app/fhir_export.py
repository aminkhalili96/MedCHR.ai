"""
Bulk Import/Export Module

Provides FHIR bundle import/export for interoperability.

Gap Reference: I01, I02
"""

import json
from datetime import datetime
from typing import Generator
from uuid import uuid4


def generate_fhir_bundle(resources: list) -> dict:
    """
    Create a FHIR Bundle from a list of resources.
    """
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "id": str(uuid4()),
        "meta": {
            "lastUpdated": datetime.utcnow().isoformat() + "Z"
        },
        "entry": [
            {
                "fullUrl": f"urn:uuid:{resource.get('id', uuid4())}",
                "resource": resource
            }
            for resource in resources
        ]
    }


def patient_to_fhir(patient: dict) -> dict:
    """Convert internal patient to FHIR Patient resource."""
    return {
        "resourceType": "Patient",
        "id": str(patient.get("id")),
        "meta": {
            "lastUpdated": datetime.utcnow().isoformat() + "Z"
        },
        "identifier": [
            {
                "use": "official",
                "system": "urn:medchr:patient-id",
                "value": str(patient.get("id"))
            }
        ],
        "name": [
            {
                "use": "official",
                "text": patient.get("full_name"),
                "family": patient.get("full_name", "").split()[-1] if patient.get("full_name") else None,
                "given": patient.get("full_name", "").split()[:-1] if patient.get("full_name") else []
            }
        ],
        "birthDate": str(patient.get("dob")) if patient.get("dob") else None
    }


def observation_to_fhir(lab: dict, patient_id: str) -> dict:
    """Convert internal lab result to FHIR Observation."""
    observation = {
        "resourceType": "Observation",
        "id": str(uuid4()),
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory"
                    }
                ]
            }
        ],
        "code": {
            "text": lab.get("test_name")
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        }
    }
    
    # Add LOINC code if available
    if lab.get("loinc"):
        observation["code"]["coding"] = [
            {
                "system": "http://loinc.org",
                "code": lab["loinc"].get("code"),
                "display": lab["loinc"].get("name")
            }
        ]
    
    # Add value
    value = lab.get("value")
    unit = lab.get("unit", "")
    
    try:
        observation["valueQuantity"] = {
            "value": float(value),
            "unit": unit,
            "system": "http://unitsofmeasure.org"
        }
    except (ValueError, TypeError):
        observation["valueString"] = str(value)
    
    # Add reference range
    if lab.get("reference_range"):
        observation["referenceRange"] = [
            {"text": lab["reference_range"]}
        ]
    
    # Add interpretation
    flag = lab.get("flag", "")
    if flag:
        interpretation_map = {
            "H": {"code": "H", "display": "High"},
            "L": {"code": "L", "display": "Low"},
            "C": {"code": "A", "display": "Abnormal"},
            "N": {"code": "N", "display": "Normal"}
        }
        if flag in interpretation_map:
            observation["interpretation"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            **interpretation_map[flag]
                        }
                    ]
                }
            ]
    
    return observation


def medication_statement_to_fhir(med: dict, patient_id: str) -> dict:
    """Convert internal medication to FHIR MedicationStatement."""
    statement = {
        "resourceType": "MedicationStatement",
        "id": str(uuid4()),
        "status": "active" if med.get("status") != "discontinued" else "stopped",
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "medicationCodeableConcept": {
            "text": med.get("name")
        }
    }
    
    # Add RxNorm code if available
    if med.get("rxnorm"):
        statement["medicationCodeableConcept"]["coding"] = [
            {
                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": med["rxnorm"].get("rxcui"),
                "display": med["rxnorm"].get("name")
            }
        ]
    
    # Add dosage
    if med.get("dosage") or med.get("frequency"):
        statement["dosage"] = [
            {
                "text": f"{med.get('dosage', '')} {med.get('frequency', '')}".strip(),
                "route": {
                    "text": med.get("route", "oral")
                }
            }
        ]
    
    return statement


def condition_to_fhir(dx: dict, patient_id: str) -> dict:
    """Convert internal diagnosis to FHIR Condition."""
    condition = {
        "resourceType": "Condition",
        "id": str(uuid4()),
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "code": {
            "text": dx.get("condition")
        }
    }
    
    # Add SNOMED code if available
    if dx.get("snomed"):
        condition["code"]["coding"] = [
            {
                "system": "http://snomed.info/sct",
                "code": dx["snomed"].get("code"),
                "display": dx["snomed"].get("display")
            }
        ]
    
    # Add clinical status
    status_map = {
        "active": {"code": "active", "display": "Active"},
        "resolved": {"code": "resolved", "display": "Resolved"},
        "negated": {"code": "refuted", "display": "Refuted"},
        "historical": {"code": "inactive", "display": "Inactive"}
    }
    status = dx.get("status", "active")
    if status in status_map:
        condition["clinicalStatus"] = {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    **status_map[status]
                }
            ]
        }
    
    return condition


def export_patient_bundle(patient: dict, extraction: dict) -> dict:
    """
    Export complete patient data as FHIR Bundle.
    """
    resources = []
    
    # Add patient
    patient_id = str(patient.get("id"))
    resources.append(patient_to_fhir(patient))
    
    # Add labs
    for lab in extraction.get("labs", []):
        resources.append(observation_to_fhir(lab, patient_id))
    
    # Add medications
    for med in extraction.get("medications", []):
        resources.append(medication_statement_to_fhir(med, patient_id))
    
    # Add diagnoses
    for dx in extraction.get("diagnoses", []):
        resources.append(condition_to_fhir(dx, patient_id))
    
    return generate_fhir_bundle(resources)


def export_bundle_ndjson(bundle: dict) -> Generator[str, None, None]:
    """
    Export bundle in NDJSON format (for bulk export).
    """
    for entry in bundle.get("entry", []):
        yield json.dumps(entry.get("resource", {}))


def import_fhir_bundle(bundle: dict, patient_id: str) -> dict:
    """
    Import a FHIR Bundle into internal format.
    Returns extraction-compatible dict.
    """
    result = {
        "labs": [],
        "medications": [],
        "diagnoses": []
    }
    
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        
        if resource_type == "Observation":
            lab = {
                "test_name": resource.get("code", {}).get("text", ""),
                "value": "",
                "unit": ""
            }
            
            # Get value
            if "valueQuantity" in resource:
                vq = resource["valueQuantity"]
                lab["value"] = str(vq.get("value", ""))
                lab["unit"] = vq.get("unit", "")
            elif "valueString" in resource:
                lab["value"] = resource["valueString"]
            
            # Get LOINC
            for coding in resource.get("code", {}).get("coding", []):
                if coding.get("system") == "http://loinc.org":
                    lab["loinc"] = {
                        "code": coding.get("code"),
                        "name": coding.get("display")
                    }
            
            result["labs"].append(lab)
        
        elif resource_type == "MedicationStatement":
            med = {
                "name": resource.get("medicationCodeableConcept", {}).get("text", ""),
                "status": "active" if resource.get("status") == "active" else "discontinued"
            }
            
            # Get RxNorm
            for coding in resource.get("medicationCodeableConcept", {}).get("coding", []):
                if "rxnorm" in coding.get("system", ""):
                    med["rxnorm"] = {
                        "rxcui": coding.get("code"),
                        "name": coding.get("display")
                    }
            
            # Get dosage
            if resource.get("dosage"):
                med["dosage"] = resource["dosage"][0].get("text", "")
            
            result["medications"].append(med)
        
        elif resource_type == "Condition":
            dx = {
                "condition": resource.get("code", {}).get("text", ""),
                "status": "active"
            }
            
            # Get SNOMED
            for coding in resource.get("code", {}).get("coding", []):
                if "snomed" in coding.get("system", ""):
                    dx["snomed"] = {
                        "code": coding.get("code"),
                        "display": coding.get("display")
                    }
            
            result["diagnoses"].append(dx)
    
    return result
