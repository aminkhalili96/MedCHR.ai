"""
Validation Test Script for Hospital-Grade Enhancements

This script validates the new accuracy and quality features:
1. Vision OCR (if enabled)
2. Terminology mapping (LOINC, SNOMED, RxNorm)
3. Critical value detection
4. Drug interaction checking
5. Negation detection
6. Confidence scoring
7. Hybrid RAG retrieval
"""

import json
import sys
sys.path.insert(0, ".")

from backend.app.terminology import (
    map_to_loinc, map_to_snomed, map_to_rxnorm, 
    convert_units, enrich_lab_results
)
from backend.app.alerts import (
    check_critical_values, check_drug_interactions,
    check_allergy_contraindications, run_safety_checks
)
from backend.app.extract import extract_structured


def test_terminology_mapping():
    """Test terminology services."""
    print("\n" + "="*60)
    print("TEST: Terminology Mapping")
    print("="*60)
    
    # SNOMED
    conditions = ["diabetes", "hypertension", "COPD", "heart failure"]
    print("\nSNOMED CT Mapping:")
    for condition in conditions:
        result = map_to_snomed(condition)
        if result:
            print(f"  ✓ {condition} → {result['code']} ({result['display']})")
        else:
            print(f"  ✗ {condition} → No mapping found")
    
    # RxNorm
    medications = ["metformin", "lisinopril", "atorvastatin", "aspirin"]
    print("\nRxNorm Mapping:")
    for med in medications:
        result = map_to_rxnorm(med)
        if result:
            print(f"  ✓ {med} → {result['rxcui']} ({result['name']})")
        else:
            print(f"  ✗ {med} → No mapping found")
    
    # Unit Conversion
    print("\nUnit Conversion:")
    conversions = [
        (126, "mg/dL", "mmol/L", "glucose"),
        (5.5, "mmol/L", "mg/dL", "glucose"),
        (1.2, "mg/dL", "umol/L", "creatinine"),
    ]
    for value, from_u, to_u, analyte in conversions:
        result = convert_units(value, from_u, to_u, analyte)
        if result:
            print(f"  ✓ {value} {from_u} ({analyte}) → {result} {to_u}")
        else:
            print(f"  ✗ {value} {from_u} ({analyte}) → Conversion not available")


def test_critical_values():
    """Test critical value detection."""
    print("\n" + "="*60)
    print("TEST: Critical Value Detection")
    print("="*60)
    
    # Sample labs with some critical values
    labs = [
        {"test_name": "Potassium", "value": "6.8", "unit": "mEq/L"},  # Critical HIGH
        {"test_name": "Potassium", "value": "4.5", "unit": "mEq/L"},  # Normal
        {"test_name": "Glucose", "value": "35", "unit": "mg/dL"},     # Critical LOW
        {"test_name": "Hemoglobin", "value": "6.5", "unit": "g/dL"},  # Critical LOW
        {"test_name": "Creatinine", "value": "1.3", "unit": "mg/dL"}, # Normal
        {"test_name": "Sodium", "value": "118", "unit": "mEq/L"},     # Critical LOW
    ]
    
    alerts = check_critical_values(labs)
    
    if alerts:
        print(f"\n⚠️  Found {len(alerts)} critical value alerts:")
        for alert in alerts:
            print(f"  🔴 {alert['test']}: {alert['value']} {alert['unit']} - {alert['direction']}")
            print(f"     Action: {alert['action']}")
    else:
        print("\n✓ No critical values detected")


def test_drug_interactions():
    """Test drug interaction checking."""
    print("\n" + "="*60)
    print("TEST: Drug Interaction Checking")
    print("="*60)
    
    # John Doe's medications from the mock data
    medications = [
        "Lisinopril 20mg",
        "Metformin 1000mg",
        "Atorvastatin 40mg",
        "Aspirin 81mg",
        "Carvedilol 12.5mg",
        "Warfarin 5mg",  # Adding this to test interaction
    ]
    
    interactions = check_drug_interactions(medications)
    
    if interactions:
        print(f"\n⚠️  Found {len(interactions)} drug interactions:")
        for interaction in interactions:
            print(f"  ⚡ {interaction['drug1']} + {interaction['drug2']}")
            print(f"     Severity: {interaction['severity']}")
            print(f"     Effect: {interaction['effect']}")
            print(f"     Recommendation: {interaction['recommendation']}")
    else:
        print("\n✓ No drug interactions detected")


def test_allergy_contraindications():
    """Test allergy-drug contraindication checking."""
    print("\n" + "="*60)
    print("TEST: Allergy-Drug Contraindication")
    print("="*60)
    
    allergies = ["Penicillin", "Sulfa"]
    medications = ["Amoxicillin 500mg", "Metformin 1000mg", "Sulfamethoxazole"]
    
    alerts = check_allergy_contraindications(allergies, medications)
    
    if alerts:
        print(f"\n🚨 Found {len(alerts)} contraindications:")
        for alert in alerts:
            print(f"  ⛔ {alert['allergy']} allergy vs {alert['medication']}")
            print(f"     Severity: {alert['severity']}")
            print(f"     Message: {alert['message']}")
    else:
        print("\n✓ No allergy contraindications detected")


def test_comprehensive_safety():
    """Test comprehensive safety check."""
    print("\n" + "="*60)
    print("TEST: Comprehensive Safety Check")
    print("="*60)
    
    labs = [
        {"test_name": "Potassium", "value": "6.2", "unit": "mEq/L"},
        {"test_name": "Creatinine", "value": "2.5", "unit": "mg/dL"},
    ]
    medications = ["Lisinopril", "Spironolactone", "Warfarin", "Aspirin"]
    allergies = ["Penicillin"]
    
    results = run_safety_checks(labs, medications, allergies)
    
    print(f"\nSafety Summary:")
    print(f"  Critical alerts: {results['summary']['critical_count']}")
    print(f"  High alerts: {results['summary']['high_count']}")
    print(f"  Moderate alerts: {results['summary']['moderate_count']}")
    print(f"  Requires immediate action: {results['summary']['requires_immediate_action']}")


def test_extraction_with_sample_data():
    """Test extraction on sample data (requires API key)."""
    print("\n" + "="*60)
    print("TEST: Extraction with Sample Data")
    print("="*60)
    
    sample_text = """
    LABORATORY REPORT
    Patient: John Doe
    Date: 2025-01-08
    
    BASIC METABOLIC PANEL:
    Sodium: 138 mmol/L (135-145)
    Potassium: 6.8 mmol/L (3.5-5.0) [CRITICAL HIGH]
    Creatinine: 1.3 mg/dL (0.6-1.2) [HIGH]
    
    HISTORY:
    - No diabetes (negated)
    - Denies chest pain (negated)
    - Hypertension - controlled
    
    MEDICATIONS:
    - Lisinopril 20mg PO daily
    - Metformin 1000mg PO BID
    
    ALLERGIES: Penicillin (rash)
    """
    
    try:
        from backend.app.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            print("  ⚠️  Skipping extraction test - no API key configured")
            return
        
        print("  Running extraction...")
        result = extract_structured(sample_text, enrich=True)
        
        print(f"\n  Extracted {len(result.get('labs', []))} labs")
        print(f"  Extracted {len(result.get('medications', []))} medications")
        print(f"  Extracted {len(result.get('diagnoses', []))} diagnoses")
        
        if result.get('safety_alerts'):
            print(f"\n  ⚠️  Safety alerts: {len(result['safety_alerts'])}")
            for alert in result['safety_alerts'][:3]:
                print(f"     - {alert.get('test', alert.get('drug1', 'Unknown'))}: {alert.get('severity', 'N/A')}")
        
        # Check negation detection
        negated = [d for d in result.get('diagnoses', []) if d.get('status') == 'negated']
        if negated:
            print(f"\n  ✓ Negation detection working: {len(negated)} negated conditions found")
        
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")


def main():
    """Run all validation tests."""
    print("\n" + "#"*60)
    print("# MedCHR.ai Hospital-Grade Validation Tests")
    print("#"*60)
    
    test_terminology_mapping()
    test_critical_values()
    test_drug_interactions()
    test_allergy_contraindications()
    test_comprehensive_safety()
    test_extraction_with_sample_data()
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
