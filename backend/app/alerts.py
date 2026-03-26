"""
Clinical Alerts Engine for critical value detection and medication safety.

Provides:
- Critical lab value detection (life-threatening ranges)
- Abnormal trending detection
- Drug-drug interaction warnings
- Allergy-drug contraindication checks

Gap References: C01-C10

DISCLAIMER: Reference ranges are general adult values derived from common
clinical laboratory references (Tietz Clinical Guide to Laboratory Tests,
4th Edition; CLSI guidelines). These ranges do NOT account for age, sex,
pregnancy, ethnicity, or assay-specific variations. Always verify against
your laboratory's own published reference ranges before making clinical
decisions. This module is a decision-support tool — NOT a substitute for
clinical judgement.

Drug interactions listed here are an illustrative subset, NOT a comprehensive
database. Use authoritative drug interaction databases (e.g., Lexicomp,
Micromedex, Clinical Pharmacology) for production clinical decisions.
"""

from typing import Optional
from datetime import datetime, timedelta


# ============================================================================
# Critical Lab Value Ranges
# Source: General adult ranges — see module docstring for limitations
# ============================================================================

# Values outside these ranges require immediate clinical attention
CRITICAL_RANGES = {
    # Electrolytes
    "potassium": {
        "low": 2.5, "high": 6.5, 
        "unit": "mEq/L",
        "action": "Immediate ECG and treatment required"
    },
    "sodium": {
        "low": 120, "high": 160,
        "unit": "mEq/L", 
        "action": "Evaluate fluid status and neurological symptoms"
    },
    "calcium": {
        "low": 6.0, "high": 13.0,
        "unit": "mg/dL",
        "action": "Check ionized calcium; cardiac monitoring"
    },
    "magnesium": {
        "low": 1.0, "high": 4.0,
        "unit": "mg/dL",
        "action": "Monitor for cardiac arrhythmias"
    },
    
    # Glucose
    "glucose": {
        "low": 40, "high": 500,
        "unit": "mg/dL",
        "action": "Hypoglycemia: give dextrose; Hyperglycemia: check ketones"
    },
    
    # Renal
    "creatinine": {
        "low": None, "high": 10.0,
        "unit": "mg/dL",
        "action": "Evaluate for dialysis; adjust medications"
    },
    "bun": {
        "low": None, "high": 100,
        "unit": "mg/dL",
        "action": "Evaluate renal function and hydration"
    },
    
    # Hematology
    "hemoglobin": {
        "low": 7.0, "high": 20.0,
        "unit": "g/dL",
        "action": "Low: consider transfusion; High: evaluate polycythemia"
    },
    "hematocrit": {
        "low": 20, "high": 60,
        "unit": "%",
        "action": "Evaluate for anemia or polycythemia"
    },
    "platelets": {
        "low": 20, "high": 1000,
        "unit": "×10³/µL",
        "action": "Low: bleeding precautions; High: thrombosis risk"
    },
    "wbc": {
        "low": 1.0, "high": 30.0,
        "unit": "×10³/µL",
        "action": "Low: infection risk; High: evaluate for leukocytosis cause"
    },
    
    # Coagulation
    "inr": {
        "low": None, "high": 5.0,
        "unit": "",
        "action": "High: hold anticoagulation, consider vitamin K"
    },
    "ptt": {
        "low": None, "high": 100,
        "unit": "seconds",
        "action": "High: bleeding risk, check for heparin effect"
    },
    
    # Cardiac
    "troponin": {
        "low": None, "high": 0.04,
        "unit": "ng/mL",
        "action": "Elevated: evaluate for ACS, serial monitoring"
    },
    
    # Liver
    "ast": {
        "low": None, "high": 1000,
        "unit": "U/L",
        "action": "Severe elevation: evaluate for acute liver injury"
    },
    "alt": {
        "low": None, "high": 1000,
        "unit": "U/L",
        "action": "Severe elevation: evaluate for acute liver injury"
    },
    "bilirubin": {
        "low": None, "high": 15.0,
        "unit": "mg/dL",
        "action": "Evaluate for liver failure or biliary obstruction"
    },
    
    # Blood Gases
    "ph": {
        "low": 7.2, "high": 7.6,
        "unit": "",
        "action": "Acidosis/Alkalosis: immediate intervention required"
    },
    "pco2": {
        "low": 20, "high": 70,
        "unit": "mmHg",
        "action": "Respiratory failure evaluation"
    },
    "po2": {
        "low": 40, "high": None,
        "unit": "mmHg",
        "action": "Low: evaluate oxygenation, consider supplemental O2"
    },
}


def parse_numeric(value: str) -> Optional[float]:
    """Parse a numeric value from string, handling common formats."""
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # Clean the string
    cleaned = str(value).strip()
    
    # Remove common prefixes/suffixes
    cleaned = cleaned.replace("<", "").replace(">", "").replace("=", "")
    cleaned = cleaned.replace(",", "")
    
    # Handle ranges like "120-140" - take the average
    if "-" in cleaned and not cleaned.startswith("-"):
        parts = cleaned.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            pass
    
    try:
        return float(cleaned)
    except ValueError:
        return None


# Abnormal (warning-level) ranges for common tests — flags values outside
# the normal reference range even when not immediately life-threatening.
ABNORMAL_RANGES = {
    "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL", "action": "Review for anemia or polycythemia"},
    "hba1c": {"low": None, "high": 6.5, "unit": "%", "action": "Evaluate glycemic control"},
    "fasting glucose": {"low": 70, "high": 100, "unit": "mg/dL", "action": "Assess for diabetes or hypoglycemia"},
    "glucose": {"low": 70, "high": 140, "unit": "mg/dL", "action": "Assess glucose regulation"},
    "total cholesterol": {"low": None, "high": 200, "unit": "mg/dL", "action": "Lipid management review"},
    "ldl": {"low": None, "high": 130, "unit": "mg/dL", "action": "Consider statin therapy"},
    "hdl": {"low": 40, "high": None, "unit": "mg/dL", "action": "Low HDL: cardiovascular risk factor"},
    "triglycerides": {"low": None, "high": 150, "unit": "mg/dL", "action": "Lifestyle and dietary review"},
    "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL", "action": "Evaluate renal function"},
    "cea": {"low": None, "high": 5.0, "unit": "ng/mL", "action": "Elevated tumor marker — further evaluation"},
    "potassium": {"low": 3.5, "high": 5.0, "unit": "mEq/L", "action": "Monitor electrolyte balance"},
    "sodium": {"low": 136, "high": 145, "unit": "mEq/L", "action": "Evaluate fluid balance"},
}


def check_critical_values(labs: list[dict]) -> list[dict]:
    """
    Check lab results against critical value thresholds AND abnormal ranges.
    
    Args:
        labs: List of lab results with test_name, value, unit
        
    Returns:
        List of critical/warning value alerts
    """
    alerts = []
    seen_tests = set()
    
    for lab in labs:
        test_name = str(lab.get("test_name", "")).lower()
        value = parse_numeric(lab.get("value"))
        
        if value is None:
            continue
        
        # --- Critical ranges (life-threatening) ---
        for critical_test, ranges in CRITICAL_RANGES.items():
            if critical_test in test_name:
                is_critical = False
                direction = None
                
                if ranges["low"] is not None and value < ranges["low"]:
                    is_critical = True
                    direction = "CRITICALLY LOW"
                elif ranges["high"] is not None and value > ranges["high"]:
                    is_critical = True
                    direction = "CRITICALLY HIGH"
                
                if is_critical:
                    alerts.append({
                        "test": lab.get("test_name"),
                        "value": value,
                        "unit": lab.get("unit", ranges.get("unit", "")),
                        "threshold_low": ranges["low"],
                        "threshold_high": ranges["high"],
                        "severity": "CRITICAL",
                        "direction": direction,
                        "action": ranges["action"],
                        "timestamp": datetime.now().isoformat()
                    })
                    seen_tests.add(test_name)
                break
        
        # --- Abnormal ranges (warning-level) ---
        if test_name not in seen_tests:
            for abnormal_test, ranges in ABNORMAL_RANGES.items():
                if abnormal_test in test_name:
                    is_abnormal = False
                    direction = None
                    
                    if ranges["low"] is not None and value < ranges["low"]:
                        is_abnormal = True
                        direction = "LOW"
                    elif ranges["high"] is not None and value > ranges["high"]:
                        is_abnormal = True
                        direction = "HIGH"
                    
                    if is_abnormal:
                        alerts.append({
                            "test": lab.get("test_name"),
                            "value": value,
                            "unit": lab.get("unit", ranges.get("unit", "")),
                            "severity": "WARNING",
                            "direction": direction,
                            "action": ranges["action"],
                            "timestamp": datetime.now().isoformat()
                        })
                        seen_tests.add(test_name)
                    break
    
    return alerts


def check_abnormal_trend(current_labs: list[dict], historical_labs: list[dict], 
                         threshold_pct: float = 50) -> list[dict]:
    """
    Detect significant worsening trends in lab values.
    
    Args:
        current_labs: Current lab results
        historical_labs: Previous lab results (last 7 days recommended)
        threshold_pct: Percentage change to flag as concerning
        
    Returns:
        List of trending alerts
    """
    alerts = []
    
    # Group historical by test name
    historical_by_test = {}
    for lab in historical_labs:
        test = lab.get("test_name", "").lower()
        if test not in historical_by_test:
            historical_by_test[test] = []
        historical_by_test[test].append(lab)
    
    for current in current_labs:
        test = current.get("test_name", "").lower()
        current_value = parse_numeric(current.get("value"))
        
        if current_value is None or test not in historical_by_test:
            continue
        
        # Get the most recent historical value
        prev_labs = historical_by_test[test]
        if not prev_labs:
            continue
        
        prev_value = parse_numeric(prev_labs[-1].get("value"))
        if prev_value is None or prev_value == 0:
            continue
        
        # Calculate percent change
        pct_change = ((current_value - prev_value) / prev_value) * 100
        
        if abs(pct_change) >= threshold_pct:
            direction = "INCREASING" if pct_change > 0 else "DECREASING"
            alerts.append({
                "test": current.get("test_name"),
                "current_value": current_value,
                "previous_value": prev_value,
                "percent_change": round(pct_change, 1),
                "direction": direction,
                "severity": "WARNING",
                "message": f"{direction} trend: {abs(round(pct_change, 1))}% change"
            })
    
    return alerts


# ============================================================================
# Drug Interaction Checking
# ============================================================================

# High-risk drug interactions — ILLUSTRATIVE SUBSET ONLY.
# NOT a comprehensive database. ~15 interactions listed; production systems
# should integrate a full drug interaction API (e.g., RxNorm, Lexicomp).
DRUG_INTERACTIONS = {
    ("warfarin", "aspirin"): {
        "severity": "HIGH",
        "effect": "Increased bleeding risk",
        "recommendation": "Monitor INR closely; consider GI prophylaxis"
    },
    ("metformin", "contrast"): {
        "severity": "HIGH",
        "effect": "Risk of lactic acidosis",
        "recommendation": "Hold metformin 48h before and after contrast"
    },
    ("ace inhibitor", "potassium"): {
        "severity": "MODERATE",
        "effect": "Risk of hyperkalemia",
        "recommendation": "Monitor potassium levels"
    },
    ("lisinopril", "spironolactone"): {
        "severity": "MODERATE",
        "effect": "Risk of hyperkalemia",
        "recommendation": "Monitor potassium levels"
    },
    ("ssri", "maoi"): {
        "severity": "CRITICAL",
        "effect": "Serotonin syndrome risk",
        "recommendation": "Do not combine; wash-out period required"
    },
    ("simvastatin", "amiodarone"): {
        "severity": "HIGH",
        "effect": "Increased risk of rhabdomyolysis",
        "recommendation": "Limit simvastatin to 20mg daily"
    },
    ("methotrexate", "nsaid"): {
        "severity": "HIGH",
        "effect": "Increased methotrexate toxicity",
        "recommendation": "Avoid combination or monitor closely"
    },
}


def check_drug_interactions(medications: list[str]) -> list[dict]:
    """
    Check for known drug-drug interactions.
    
    Args:
        medications: List of medication names
        
    Returns:
        List of interaction alerts
    """
    alerts = []
    normalized_meds = [m.lower().strip() for m in medications if m]
    
    # Check each pair
    for i, med1 in enumerate(normalized_meds):
        for med2 in normalized_meds[i+1:]:
            # Check both orderings
            for key, interaction in DRUG_INTERACTIONS.items():
                if (key[0] in med1 and key[1] in med2) or \
                   (key[1] in med1 and key[0] in med2):
                    alerts.append({
                        "drug1": med1,
                        "drug2": med2,
                        "severity": interaction["severity"],
                        "effect": interaction["effect"],
                        "recommendation": interaction["recommendation"]
                    })
                    break
    
    return alerts


# ============================================================================
# Allergy-Drug Contraindications
# ============================================================================

ALLERGY_DRUG_CONTRAINDICATIONS = {
    "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "nafcillin"],
    "sulfa": ["sulfamethoxazole", "trimethoprim-sulfamethoxazole", "sulfasalazine"],
    "aspirin": ["nsaid", "ibuprofen", "naproxen", "ketorolac"],
    "codeine": ["morphine", "hydrocodone", "oxycodone"],
    "nsaid": ["ibuprofen", "naproxen", "ketorolac", "meloxicam", "celecoxib"],
    "ace inhibitor": ["lisinopril", "enalapril", "ramipril", "benazepril"],
}


def check_allergy_contraindications(allergies: list[str], medications: list[str]) -> list[dict]:
    """
    Check if any prescribed medications contraindicated by patient allergies.
    """
    alerts = []
    
    normalized_allergies = [a.lower().strip() for a in allergies if a]
    normalized_meds = [m.lower().strip() for m in medications if m]
    
    for allergy in normalized_allergies:
        # Direct match
        for med in normalized_meds:
            if allergy in med or med in allergy:
                alerts.append({
                    "allergy": allergy,
                    "medication": med,
                    "severity": "CRITICAL",
                    "message": f"Patient allergic to {allergy}; {med} is contraindicated"
                })
        
        # Class contraindications
        for allergy_class, drugs in ALLERGY_DRUG_CONTRAINDICATIONS.items():
            if allergy_class in allergy:
                for med in normalized_meds:
                    for drug in drugs:
                        if drug in med:
                            alerts.append({
                                "allergy": allergy,
                                "medication": med,
                                "severity": "HIGH",
                                "message": f"Patient allergic to {allergy_class} class; {med} may be contraindicated"
                            })
    
    return alerts


# ============================================================================
# Comprehensive Safety Check
# ============================================================================

def run_safety_checks(
    labs: list[dict],
    medications: list[str],
    allergies: list[str],
    historical_labs: list[dict] = None
) -> dict:
    """
    Run all clinical safety checks and return consolidated alerts.
    """
    results = {
        "critical_values": check_critical_values(labs),
        "drug_interactions": check_drug_interactions(medications),
        "allergy_alerts": check_allergy_contraindications(allergies, medications),
        "trends": [],
        "summary": {
            "critical_count": 0,
            "high_count": 0,
            "moderate_count": 0,
            "requires_immediate_action": False
        }
    }
    
    if historical_labs:
        results["trends"] = check_abnormal_trend(labs, historical_labs)
    
    # Count by severity
    all_alerts = (
        results["critical_values"] + 
        results["drug_interactions"] + 
        results["allergy_alerts"]
    )
    
    for alert in all_alerts:
        severity = alert.get("severity", "").upper()
        if severity == "CRITICAL":
            results["summary"]["critical_count"] += 1
            results["summary"]["requires_immediate_action"] = True
        elif severity == "HIGH":
            results["summary"]["high_count"] += 1
        elif severity == "MODERATE":
            results["summary"]["moderate_count"] += 1
    
    return results
