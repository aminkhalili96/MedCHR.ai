"""
Data Quality Engine

Provides data validation, missing data detection, and quality metrics.

Gap Reference: DQ01-DQ05
"""

from typing import Any, Optional
from datetime import datetime, date
from enum import Enum


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DataQualityRule:
    """Base class for data quality rules."""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.WARNING):
        self.name = name
        self.severity = severity
    
    def validate(self, data: dict) -> Optional[dict]:
        """Override in subclass. Return None if valid, or dict with issue details."""
        raise NotImplementedError


class RequiredFieldRule(DataQualityRule):
    """Check for required fields."""
    
    def __init__(self, field_path: str, entity_type: str):
        super().__init__(f"required_{field_path}", ValidationSeverity.ERROR)
        self.field_path = field_path
        self.entity_type = entity_type
    
    def validate(self, data: dict) -> Optional[dict]:
        value = self._get_nested(data, self.field_path)
        if value is None or (isinstance(value, str) and not value.strip()):
            return {
                "rule": self.name,
                "severity": self.severity.value,
                "message": f"Missing required field: {self.field_path}",
                "entity_type": self.entity_type,
                "field": self.field_path
            }
        return None
    
    def _get_nested(self, data: dict, path: str) -> Any:
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value


class NumericRangeRule(DataQualityRule):
    """Check if numeric value is within expected range."""
    
    def __init__(self, field: str, min_val: float = None, max_val: float = None):
        super().__init__(f"range_{field}", ValidationSeverity.WARNING)
        self.field = field
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, data: dict) -> Optional[dict]:
        value = data.get(self.field)
        if value is None:
            return None
        
        try:
            num_val = float(str(value).replace(",", ""))
        except ValueError:
            return {
                "rule": self.name,
                "severity": self.severity.value,
                "message": f"Non-numeric value in {self.field}: {value}",
                "field": self.field,
                "value": value
            }
        
        if self.min_val is not None and num_val < self.min_val:
            return {
                "rule": self.name,
                "severity": self.severity.value,
                "message": f"Value below minimum: {self.field} = {num_val} (min: {self.min_val})",
                "field": self.field,
                "value": num_val
            }
        
        if self.max_val is not None and num_val > self.max_val:
            return {
                "rule": self.name,
                "severity": self.severity.value,
                "message": f"Value above maximum: {self.field} = {num_val} (max: {self.max_val})",
                "field": self.field,
                "value": num_val
            }
        
        return None


class DateValidityRule(DataQualityRule):
    """Check if date is valid and reasonable."""
    
    def __init__(self, field: str):
        super().__init__(f"date_{field}", ValidationSeverity.WARNING)
        self.field = field
    
    def validate(self, data: dict) -> Optional[dict]:
        value = data.get(self.field)
        if not value:
            return None
        
        try:
            if isinstance(value, str):
                # Try common formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                    try:
                        parsed = datetime.strptime(value, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    return {
                        "rule": self.name,
                        "severity": self.severity.value,
                        "message": f"Invalid date format: {value}",
                        "field": self.field,
                        "value": value
                    }
            elif isinstance(value, (datetime, date)):
                parsed = value if isinstance(value, date) else value.date()
            else:
                return None
            
            # Check if date is reasonable (not future, not too old)
            today = date.today()
            if parsed > today:
                return {
                    "rule": self.name,
                    "severity": ValidationSeverity.WARNING.value,
                    "message": f"Future date: {value}",
                    "field": self.field,
                    "value": str(value)
                }
            
            # Check if too old (>120 years)
            years_old = (today - parsed).days / 365
            if years_old > 120:
                return {
                    "rule": self.name,
                    "severity": ValidationSeverity.WARNING.value,
                    "message": f"Date too old (>120 years): {value}",
                    "field": self.field,
                    "value": str(value)
                }
            
        except Exception as e:
            return {
                "rule": self.name,
                "severity": self.severity.value,
                "message": f"Date validation error: {str(e)}",
                "field": self.field,
                "value": str(value)
            }
        
        return None


# Standard validation rules for medical data
PATIENT_RULES = [
    RequiredFieldRule("full_name", "patient"),
    DateValidityRule("dob"),
]

LAB_RULES = [
    RequiredFieldRule("test_name", "lab"),
    RequiredFieldRule("value", "lab"),
    NumericRangeRule("value", min_val=-1000, max_val=100000),
    DateValidityRule("date"),
]

MEDICATION_RULES = [
    RequiredFieldRule("name", "medication"),
]

DIAGNOSIS_RULES = [
    RequiredFieldRule("condition", "diagnosis"),
]


class DataQualityEngine:
    """
    Main engine for running data quality checks.
    """
    
    def __init__(self):
        self.rules = {
            "patient": PATIENT_RULES,
            "lab": LAB_RULES,
            "medication": MEDICATION_RULES,
            "diagnosis": DIAGNOSIS_RULES,
        }
    
    def validate_extraction(self, extraction: dict) -> dict:
        """
        Validate an extraction result and return quality report.
        """
        issues = []
        
        # Validate labs
        for lab in extraction.get("labs", []):
            for rule in self.rules["lab"]:
                issue = rule.validate(lab)
                if issue:
                    issues.append(issue)
        
        # Validate medications
        for med in extraction.get("medications", []):
            for rule in self.rules["medication"]:
                issue = rule.validate(med)
                if issue:
                    issues.append(issue)
        
        # Validate diagnoses
        for dx in extraction.get("diagnoses", []):
            for rule in self.rules["diagnosis"]:
                issue = rule.validate(dx)
                if issue:
                    issues.append(issue)
        
        # Calculate metrics
        error_count = sum(1 for i in issues if i["severity"] == "error")
        warning_count = sum(1 for i in issues if i["severity"] == "warning")
        
        # Calculate completeness
        completeness = self._calculate_completeness(extraction)
        
        return {
            "is_valid": error_count == 0,
            "issues": issues,
            "metrics": {
                "error_count": error_count,
                "warning_count": warning_count,
                "total_issues": len(issues),
                "completeness_score": completeness,
                "quality_score": self._calculate_quality_score(issues, completeness)
            }
        }
    
    def _calculate_completeness(self, extraction: dict) -> float:
        """Calculate data completeness score."""
        expected_sections = ["labs", "medications", "diagnoses"]
        present = sum(1 for s in expected_sections if extraction.get(s))
        
        # Also check for key fields in labs
        lab_completeness = 0
        labs = extraction.get("labs", [])
        if labs:
            complete_labs = sum(
                1 for lab in labs 
                if lab.get("test_name") and lab.get("value") and lab.get("unit")
            )
            lab_completeness = complete_labs / len(labs) if labs else 0
        
        section_score = present / len(expected_sections)
        overall = (section_score + lab_completeness) / 2 if labs else section_score
        
        return round(overall, 2)
    
    def _calculate_quality_score(self, issues: list, completeness: float) -> float:
        """Calculate overall quality score (0-1)."""
        # Start with completeness
        score = completeness
        
        # Deduct for issues
        for issue in issues:
            if issue["severity"] == "error":
                score -= 0.1
            elif issue["severity"] == "warning":
                score -= 0.02
        
        return max(0, min(1, round(score, 2)))


def check_missing_data(extraction: dict) -> list:
    """
    Identify missing data that should be present.
    
    Gap Reference: DQ02
    """
    missing = []
    
    # Check for empty sections
    sections = [
        ("labs", "No lab results extracted"),
        ("medications", "No medications extracted"),
        ("diagnoses", "No diagnoses extracted"),
    ]
    
    for section, message in sections:
        if not extraction.get(section):
            missing.append({
                "section": section,
                "message": message,
                "severity": "info"
            })
    
    # Check for incomplete labs
    for i, lab in enumerate(extraction.get("labs", [])):
        if not lab.get("unit"):
            missing.append({
                "section": "labs",
                "index": i,
                "message": f"Missing unit for {lab.get('test_name', 'Unknown test')}",
                "severity": "warning"
            })
        if not lab.get("reference_range"):
            missing.append({
                "section": "labs",
                "index": i,
                "message": f"Missing reference range for {lab.get('test_name', 'Unknown test')}",
                "severity": "info"
            })
    
    return missing


def track_manual_override(
    entity_type: str,
    entity_id: str,
    field: str,
    original_value: Any,
    corrected_value: Any,
    user_id: str,
    conn
) -> str:
    """
    Track when a user manually corrects extracted data.
    
    Gap Reference: DQ04
    """
    from psycopg.types.json import Json
    
    result = conn.execute("""
        INSERT INTO data_corrections (
            entity_type, entity_id, field, 
            original_value, corrected_value, 
            corrected_by, corrected_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        RETURNING id
    """, (
        entity_type, entity_id, field,
        Json(original_value), Json(corrected_value),
        user_id
    )).fetchone()
    
    return str(result["id"])


# Singleton instance
data_quality_engine = DataQualityEngine()
