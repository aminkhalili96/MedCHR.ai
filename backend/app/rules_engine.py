"""
Rules Engine - Gap 2: Domain-Specific Knowledge Base
Applies clinical rules for Functional Medicine pattern recognition.
"""
import json
from pathlib import Path
from functools import lru_cache
from typing import Any

DATA_PATH = Path(__file__).parent.parent / "data" / "clinical_rules.json"


@lru_cache(maxsize=1)
def _load_rules() -> dict:
    """Load clinical rules data."""
    if not DATA_PATH.exists():
        return {"rules": []}
    with open(DATA_PATH) as f:
        return json.load(f)


def _get_lab_value(labs: list[dict], test_name: str) -> float | None:
    """Extract a numeric lab value by test name."""
    test_name_lower = test_name.lower()
    for lab in labs:
        lab_name = (lab.get("test") or lab.get("test_name") or lab.get("canonical_name") or "").lower()
        if test_name_lower in lab_name or lab_name in test_name_lower:
            value = lab.get("value", "")
            try:
                # Handle values like "7.2 g/dL" or "<20"
                numeric = ''.join(c for c in str(value) if c.isdigit() or c == '.')
                return float(numeric) if numeric else None
            except (ValueError, TypeError):
                return None
    return None


def _check_gene(genetics: dict, gene_name: str, variant_contains: str = None) -> bool:
    """Check if patient has a specific gene variant."""
    findings = genetics.get("findings", []) if genetics else []
    for finding in findings:
        if finding.get("gene", "").upper() == gene_name.upper():
            if variant_contains:
                variant = finding.get("variant", "")
                if variant_contains.lower() in variant.lower():
                    return True
            else:
                return True
    return False


def _evaluate_condition(condition: dict, labs: list[dict], genetics: dict) -> bool:
    """Evaluate a single rule condition."""
    # Gene-based condition
    if "gene" in condition:
        gene_match = _check_gene(genetics, condition["gene"], condition.get("variant_contains"))
        if not gene_match:
            return False
        
        # Check additional lab condition if present
        if "and_lab" in condition:
            and_lab = condition["and_lab"]
            lab_value = _get_lab_value(labs, and_lab["test"])
            if lab_value is None:
                return False
            op = and_lab["operator"]
            target = and_lab["value"]
            if op == ">" and not lab_value > target:
                return False
            if op == "<" and not lab_value < target:
                return False
            if op == ">=" and not lab_value >= target:
                return False
            if op == "<=" and not lab_value <= target:
                return False
        
        return True
    
    # Lab-based condition
    if "lab" in condition:
        lab_value = _get_lab_value(labs, condition["lab"])
        if lab_value is None:
            return False
        
        op = condition["operator"]
        target = condition["value"]
        
        if op == ">" and not lab_value > target:
            return False
        if op == "<" and not lab_value < target:
            return False
        if op == ">=" and not lab_value >= target:
            return False
        if op == "<=" and not lab_value <= target:
            return False
        
        # Check "and_lab" if present
        if "and_lab" in condition:
            and_lab = condition["and_lab"]
            and_value = _get_lab_value(labs, and_lab["test"])
            if and_value is None:
                return False
            and_op = and_lab["operator"]
            and_target = and_lab["value"]
            if and_op == ">" and not and_value > and_target:
                return False
            if and_op == "<" and not and_value < and_target:
                return False
            if and_op == "normal":
                pass  # Assume normal if present
        
        # Check "or_lab" if present
        if "or_lab" in condition:
            or_lab = condition["or_lab"]
            or_value = _get_lab_value(labs, or_lab["test"])
            or_target = or_lab["value"]
            or_op = or_lab["operator"]
            or_match = False
            if or_value is not None:
                if or_op == ">" and or_value > or_target:
                    or_match = True
                if or_op == "<" and or_value < or_target:
                    or_match = True
            # For "or" conditions, the main condition already passed, so this is additive
        
        return True
    
    return False


def evaluate_rules(labs: list[dict], genetics: dict) -> list[dict]:
    """
    Evaluate all clinical rules against patient data.
    
    Args:
        labs: List of lab results
        genetics: Patient's genetics data
        
    Returns:
        List of triggered rules with recommendations.
    """
    data = _load_rules()
    triggered = []
    
    for rule in data.get("rules", []):
        condition = rule.get("condition", {})
        if _evaluate_condition(condition, labs, genetics):
            triggered.append({
                "rule_id": rule.get("id", ""),
                "recommendation": rule.get("recommendation", ""),
                "condition": condition
            })
    
    return triggered


def format_rules_for_chr(labs: list[dict], genetics: dict) -> str:
    """
    Format triggered rules as clinical pearls for CHR.
    
    Returns:
        Markdown-formatted string for the CHR.
    """
    triggered = evaluate_rules(labs, genetics)
    
    if not triggered:
        return ""
    
    lines = ["## Clinical Insights (Knowledge Base)", ""]
    lines.append("*The following insights are generated based on pattern matching against clinical rules:*")
    lines.append("")
    
    for i, rule in enumerate(triggered, 1):
        lines.append(f"{i}. **{rule['rule_id'].replace('_', ' ').title()}**")
        lines.append(f"   {rule['recommendation']}")
        lines.append("")
    
    return "\n".join(lines)
