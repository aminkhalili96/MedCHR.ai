import json
from typing import Dict, Any, List

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .llm_gateway import create_chat_completion, redact_if_enabled
from .schemas import ExtractionData


# Enhanced extraction prompt with negation detection and confidence scoring
EXTRACTION_PROMPT = """You are an expert medical data extraction specialist. Extract structured clinical data from the provided text with HIGH PRECISION.

## CRITICAL RULES:

### 1. NEGATION DETECTION (VERY IMPORTANT)
Pay careful attention to NEGATION words. If a condition is NEGATED, set status to "negated":
- "No diabetes" → {"condition": "diabetes", "status": "negated"}
- "Denies chest pain" → {"condition": "chest pain", "status": "negated"}
- "Negative for HIV" → {"condition": "HIV", "status": "negated"}
- "Without fever" → {"condition": "fever", "status": "negated"}
- "History of HTN" → {"condition": "hypertension", "status": "active"}

### 2. CONFIDENCE SCORING
For EACH extracted value, provide a confidence score (0.0-1.0):
- 1.0 = Clearly stated, high confidence
- 0.7-0.9 = Likely correct but some ambiguity
- 0.5-0.7 = Uncertain, requires review
- <0.5 = Low confidence, may be incorrect

### 3. NUMERIC PRECISION
- Extract EXACT numeric values as shown
- Preserve original units (do not convert)
- For ranges like "120-140", extract as "120-140"
- Flag abnormal values: "H" (High), "L" (Low), "C" (Critical)

### 4. DATE HANDLING
- Extract dates in YYYY-MM-DD format when possible
- If only partial date, preserve what's available

## OUTPUT SCHEMA:
{
  "labs": [
    {
      "test_name": "string",
      "value": "string (exact as shown)",
      "unit": "string",
      "flag": "H|L|N|C (High/Low/Normal/Critical)",
      "reference_range": "string",
      "date": "YYYY-MM-DD or null",
      "confidence": 0.0-1.0,
      "panel": "string (e.g., CBC, BMP, LFT)"
    }
  ],
  "medications": [
    {
      "name": "string",
      "dosage": "string",
      "frequency": "string",
      "route": "string (PO, IV, etc.)",
      "start_date": "YYYY-MM-DD or null",
      "status": "active|discontinued|prn",
      "confidence": 0.0-1.0
    }
  ],
  "diagnoses": [
    {
      "condition": "string",
      "code": "ICD-10 or SNOMED if available",
      "status": "active|resolved|negated|historical",
      "date_onset": "YYYY-MM-DD or null",
      "confidence": 0.0-1.0
    }
  ],
  "allergies": [
    {
      "substance": "string",
      "reaction": "string",
      "severity": "mild|moderate|severe",
      "confidence": 0.0-1.0
    }
  ],
  "vitals": [
    {
      "type": "BP|HR|Temp|Weight|SpO2|RR",
      "value": "string",
      "unit": "string",
      "date": "YYYY-MM-DD or null",
      "confidence": 0.0-1.0
    }
  ],
  "notes": "Brief clinical summary",
  "extraction_quality": {
    "overall_confidence": 0.0-1.0,
    "issues": ["list of any extraction issues or uncertainties"]
  }
}

If a field is not found, omit it. Be PRECISE with values - do not guess."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def extract_structured(text: str, enrich: bool = True) -> Dict[str, Any]:
    """
    Extract structured clinical data from text.
    
    Args:
        text: Raw text from document
        enrich: If True, add terminology codes and run safety checks
        
    Returns:
        Structured extraction with labs, medications, diagnoses, etc.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return ExtractionData().dict()

    # Smart chunking for long documents
    max_chars = 60000  # ~15k tokens for GPT-4
    safe_text = redact_if_enabled(text[:max_chars])

    try:
        response = create_chat_completion(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract all clinical data from this text:\n\n{safe_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        
        content = response.choices[0].message.content
        if not content:
            return ExtractionData().dict()
            
        data = json.loads(content)
        
        # Enrich with terminology codes and safety checks
        if enrich:
            data = enrich_extraction(data)
        
        return data
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return ExtractionData(notes=f"Extraction failed: {str(e)}").dict()


def enrich_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich extraction with terminology codes and safety checks.
    """
    try:
        from .terminology import enrich_lab_results, enrich_diagnoses, enrich_medications
        from .alerts import check_critical_values, check_drug_interactions, check_allergy_contraindications
        
        # Add terminology codes
        if data.get("labs"):
            data["labs"] = enrich_lab_results(data["labs"])
        
        if data.get("diagnoses"):
            data["diagnoses"] = enrich_diagnoses(data["diagnoses"])
        
        if data.get("medications"):
            data["medications"] = enrich_medications(data["medications"])
        
        # Run safety checks
        safety_alerts = []
        
        if data.get("labs"):
            critical = check_critical_values(data["labs"])
            safety_alerts.extend(critical)
        
        if data.get("medications"):
            med_names = [m.get("name", "") for m in data["medications"]]
            interactions = check_drug_interactions(med_names)
            safety_alerts.extend(interactions)
            
            # Check allergies vs medications
            if data.get("allergies"):
                allergy_names = [a.get("substance", "") for a in data["allergies"]]
                contraindications = check_allergy_contraindications(allergy_names, med_names)
                safety_alerts.extend(contraindications)
        
        if safety_alerts:
            data["safety_alerts"] = safety_alerts
            
    except ImportError as e:
        print(f"Enrichment modules not available: {e}")
    except Exception as e:
        print(f"Enrichment failed: {e}")
    
    return data
