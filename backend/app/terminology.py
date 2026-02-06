"""
Terminology Services for medical code normalization.

Provides:
- LOINC mapping for lab tests
- SNOMED CT for diagnoses
- RxNorm for medications
- Unit conversion

Gap References: T01, T02, T03, T04, T07
"""

from typing import Optional
from pgvector.psycopg import Vector

from .db import get_conn
from .embeddings import embed_texts


# ============================================================================
# LOINC Mapping (Labs)
# ============================================================================

def map_to_loinc(test_name: str, top_k: int = 3) -> list[dict]:
    """
    Find closest LOINC codes using vector similarity.
    
    Args:
        test_name: The lab test name to map (e.g., "Glucose, Fasting")
        top_k: Number of candidates to return
        
    Returns:
        List of LOINC candidates with code, name, and confidence
    """
    if not test_name:
        return []
    
    embedding = embed_texts([test_name])[0]
    vector = Vector(embedding)
    
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT l.code, l.long_common_name, l.component, l.system,
                   (le.embedding <-> %s) as distance
            FROM ref_loinc l
            JOIN ref_loinc_embeddings le ON le.code = l.code
            ORDER BY le.embedding <-> %s
            LIMIT %s
        """, (vector, vector, top_k)).fetchall()
    
    if not rows:
        return []
    
    return [
        {
            "code": r["code"],
            "name": r["long_common_name"],
            "component": r["component"],
            "system": r["system"],
            "confidence": max(0, 1 - float(r["distance"]))
        }
        for r in rows
    ]


def get_loinc_by_code(code: str) -> Optional[dict]:
    """Lookup LOINC code details."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ref_loinc WHERE code = %s",
            (code,)
        ).fetchone()
    return dict(row) if row else None


# ============================================================================
# SNOMED CT Mapping (Diagnoses)
# ============================================================================

# Common SNOMED mappings (subset for MVP)
SNOMED_MAPPINGS = {
    "diabetes": {"code": "73211009", "display": "Diabetes mellitus"},
    "diabetes mellitus": {"code": "73211009", "display": "Diabetes mellitus"},
    "type 2 diabetes": {"code": "44054006", "display": "Diabetes mellitus type 2"},
    "hypertension": {"code": "38341003", "display": "Hypertensive disorder"},
    "high blood pressure": {"code": "38341003", "display": "Hypertensive disorder"},
    "asthma": {"code": "195967001", "display": "Asthma"},
    "copd": {"code": "13645005", "display": "Chronic obstructive pulmonary disease"},
    "heart failure": {"code": "84114007", "display": "Heart failure"},
    "chronic kidney disease": {"code": "709044004", "display": "Chronic kidney disease"},
    "ckd": {"code": "709044004", "display": "Chronic kidney disease"},
    "anemia": {"code": "271737000", "display": "Anemia"},
    "hypothyroidism": {"code": "40930008", "display": "Hypothyroidism"},
    "hyperthyroidism": {"code": "34486009", "display": "Hyperthyroidism"},
    "depression": {"code": "35489007", "display": "Depressive disorder"},
    "anxiety": {"code": "48694002", "display": "Anxiety disorder"},
    "obesity": {"code": "414916001", "display": "Obesity"},
    "pneumonia": {"code": "233604007", "display": "Pneumonia"},
    "covid-19": {"code": "840539006", "display": "COVID-19"},
    "stroke": {"code": "230690007", "display": "Cerebrovascular accident"},
    "myocardial infarction": {"code": "22298006", "display": "Myocardial infarction"},
    "heart attack": {"code": "22298006", "display": "Myocardial infarction"},
}


def map_to_snomed(condition: str) -> Optional[dict]:
    """Map a condition name to SNOMED CT code."""
    if not condition:
        return None
    
    normalized = condition.lower().strip()
    
    # Direct lookup
    if normalized in SNOMED_MAPPINGS:
        return SNOMED_MAPPINGS[normalized]
    
    # Partial match
    for key, value in SNOMED_MAPPINGS.items():
        if key in normalized or normalized in key:
            return value
    
    return None


# ============================================================================
# RxNorm Mapping (Medications)
# ============================================================================

# Common RxNorm mappings (subset for MVP)
RXNORM_MAPPINGS = {
    "metformin": {"rxcui": "6809", "name": "Metformin"},
    "lisinopril": {"rxcui": "29046", "name": "Lisinopril"},
    "amlodipine": {"rxcui": "17767", "name": "Amlodipine"},
    "atorvastatin": {"rxcui": "83367", "name": "Atorvastatin"},
    "omeprazole": {"rxcui": "7646", "name": "Omeprazole"},
    "levothyroxine": {"rxcui": "10582", "name": "Levothyroxine"},
    "aspirin": {"rxcui": "1191", "name": "Aspirin"},
    "metoprolol": {"rxcui": "6918", "name": "Metoprolol"},
    "losartan": {"rxcui": "52175", "name": "Losartan"},
    "gabapentin": {"rxcui": "25480", "name": "Gabapentin"},
    "hydrochlorothiazide": {"rxcui": "5487", "name": "Hydrochlorothiazide"},
    "simvastatin": {"rxcui": "36567", "name": "Simvastatin"},
    "warfarin": {"rxcui": "11289", "name": "Warfarin"},
    "prednisone": {"rxcui": "8640", "name": "Prednisone"},
    "furosemide": {"rxcui": "4603", "name": "Furosemide"},
    "insulin": {"rxcui": "5856", "name": "Insulin"},
}


def map_to_rxnorm(medication: str) -> Optional[dict]:
    """Map a medication name to RxNorm code."""
    if not medication:
        return None
    
    normalized = medication.lower().strip()
    
    # Remove dosage info for matching
    for word in normalized.split():
        if word in RXNORM_MAPPINGS:
            return RXNORM_MAPPINGS[word]
    
    # Partial match
    for key, value in RXNORM_MAPPINGS.items():
        if key in normalized:
            return value
    
    return None


# ============================================================================
# Unit Conversion Engine
# ============================================================================

# Conversion factors: (from_unit, to_unit, analyte) -> lambda
UNIT_CONVERSIONS = {
    # Glucose
    ("mg/dl", "mmol/l", "glucose"): lambda x: round(x / 18.0, 2),
    ("mmol/l", "mg/dl", "glucose"): lambda x: round(x * 18.0, 1),
    
    # Cholesterol
    ("mg/dl", "mmol/l", "cholesterol"): lambda x: round(x / 38.67, 2),
    ("mmol/l", "mg/dl", "cholesterol"): lambda x: round(x * 38.67, 1),
    ("mg/dl", "mmol/l", "ldl"): lambda x: round(x / 38.67, 2),
    ("mg/dl", "mmol/l", "hdl"): lambda x: round(x / 38.67, 2),
    ("mg/dl", "mmol/l", "triglycerides"): lambda x: round(x / 88.57, 2),
    
    # Creatinine
    ("mg/dl", "umol/l", "creatinine"): lambda x: round(x * 88.4, 1),
    ("umol/l", "mg/dl", "creatinine"): lambda x: round(x / 88.4, 2),
    
    # BUN/Urea
    ("mg/dl", "mmol/l", "bun"): lambda x: round(x / 2.8, 2),
    ("mg/dl", "mmol/l", "urea"): lambda x: round(x / 6.0, 2),
    
    # Hemoglobin
    ("g/dl", "g/l", "hemoglobin"): lambda x: round(x * 10, 1),
    ("g/l", "g/dl", "hemoglobin"): lambda x: round(x / 10, 1),
    
    # Bilirubin
    ("mg/dl", "umol/l", "bilirubin"): lambda x: round(x * 17.1, 1),
    ("umol/l", "mg/dl", "bilirubin"): lambda x: round(x / 17.1, 2),
    
    # Vitamin D
    ("ng/ml", "nmol/l", "vitamin d"): lambda x: round(x * 2.496, 1),
    ("nmol/l", "ng/ml", "vitamin d"): lambda x: round(x / 2.496, 1),
    
    # Iron
    ("ug/dl", "umol/l", "iron"): lambda x: round(x * 0.179, 2),
    ("umol/l", "ug/dl", "iron"): lambda x: round(x / 0.179, 1),
    
    # Calcium
    ("mg/dl", "mmol/l", "calcium"): lambda x: round(x / 4.0, 2),
    ("mmol/l", "mg/dl", "calcium"): lambda x: round(x * 4.0, 1),
    
    # Potassium, Sodium (already in mEq/L = mmol/L for monovalent)
    ("meq/l", "mmol/l", "potassium"): lambda x: x,
    ("meq/l", "mmol/l", "sodium"): lambda x: x,
}


def convert_units(value: float, from_unit: str, to_unit: str, analyte: str) -> Optional[float]:
    """
    Convert a lab value between units.
    
    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g., "mg/dL")
        to_unit: Target unit (e.g., "mmol/L")
        analyte: The analyte name (e.g., "glucose")
        
    Returns:
        Converted value or None if conversion not available
    """
    if from_unit.lower() == to_unit.lower():
        return value
    
    key = (from_unit.lower(), to_unit.lower(), analyte.lower())
    
    if key in UNIT_CONVERSIONS:
        return UNIT_CONVERSIONS[key](value)
    
    # Try without analyte for generic conversions
    # (future: add more generic conversions)
    
    return None


def standardize_unit(unit: str) -> str:
    """Normalize unit representation."""
    if not unit:
        return ""
    
    unit = unit.lower().strip()
    
    # Common normalizations
    normalizations = {
        "mg/dl": "mg/dL",
        "mmol/l": "mmol/L",
        "umol/l": "µmol/L",
        "g/dl": "g/dL",
        "g/l": "g/L",
        "ng/ml": "ng/mL",
        "pg/ml": "pg/mL",
        "iu/l": "IU/L",
        "u/l": "U/L",
        "meq/l": "mEq/L",
        "cells/ul": "cells/µL",
        "10^9/l": "×10⁹/L",
        "10^12/l": "×10¹²/L",
        "%": "%",
    }
    
    return normalizations.get(unit, unit)


# ============================================================================
# Batch Processing
# ============================================================================

def enrich_lab_results(labs: list[dict]) -> list[dict]:
    """
    Enrich lab results with terminology codes and standardized units.
    """
    enriched = []
    
    for lab in labs:
        enriched_lab = dict(lab)
        
        # Add LOINC mapping
        test_name = lab.get("test_name", "")
        loinc_matches = map_to_loinc(test_name, top_k=1)
        if loinc_matches:
            enriched_lab["loinc"] = loinc_matches[0]
        
        # Standardize units
        unit = lab.get("unit", "")
        enriched_lab["unit_standardized"] = standardize_unit(unit)
        
        enriched.append(enriched_lab)
    
    return enriched


def enrich_diagnoses(diagnoses: list[dict]) -> list[dict]:
    """Enrich diagnoses with SNOMED codes."""
    enriched = []
    
    for dx in diagnoses:
        enriched_dx = dict(dx)
        
        condition = dx.get("condition", "")
        snomed = map_to_snomed(condition)
        if snomed:
            enriched_dx["snomed"] = snomed
        
        enriched.append(enriched_dx)
    
    return enriched


def enrich_medications(medications: list[dict]) -> list[dict]:
    """Enrich medications with RxNorm codes."""
    enriched = []
    
    for med in medications:
        enriched_med = dict(med)
        
        name = med.get("name", "")
        rxnorm = map_to_rxnorm(name)
        if rxnorm:
            enriched_med["rxnorm"] = rxnorm
        
        enriched.append(enriched_med)
    
    return enriched
