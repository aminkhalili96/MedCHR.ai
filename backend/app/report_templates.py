"""
Report Templates Module

Provides specialty-specific CHR templates and formatting.

Gap Reference: R07
"""

from typing import Dict, Any, List


# ============================================================================
# Base Template
# ============================================================================

class ReportTemplate:
    """Base class for CHR report templates."""
    
    name: str = "generic"
    display_name: str = "Generic Clinical Report"
    
    # Sections in order
    sections: List[str] = [
        "summary",
        "key_findings",
        "labs",
        "medications",
        "diagnoses",
        "interpretation",
        "recommendations",
        "follow_up"
    ]
    
    @classmethod
    def format_report(cls, data: Dict[str, Any]) -> str:
        """Format report data using this template."""
        parts = []
        
        for section in cls.sections:
            content = cls._format_section(section, data)
            if content:
                parts.append(content)
        
        return "\n\n".join(parts)
    
    @classmethod
    def _format_section(cls, section: str, data: Dict[str, Any]) -> str:
        """Format a single section."""
        method = getattr(cls, f"_format_{section}", None)
        if method:
            return method(data)
        
        # Default formatting
        content = data.get(section)
        if not content:
            return ""
        
        title = section.replace("_", " ").title()
        if isinstance(content, list):
            items = "\n".join(f"- {item}" for item in content)
            return f"## {title}\n\n{items}"
        return f"## {title}\n\n{content}"
    
    @classmethod
    def _format_labs(cls, data: Dict[str, Any]) -> str:
        """Format lab results as a table."""
        labs = data.get("labs", [])
        if not labs:
            return ""
        
        lines = ["## Laboratory Results", "", "| Test | Value | Range | Flag |", "|------|-------|-------|------|"]
        
        for lab in labs:
            test = lab.get("test_name", "")
            value = f"{lab.get('value', '')} {lab.get('unit', '')}".strip()
            ref = lab.get("reference_range", "")
            flag = lab.get("flag", "")
            flag_emoji = {"H": "⬆️", "L": "⬇️", "C": "🔴"}.get(flag, "")
            lines.append(f"| {test} | {value} | {ref} | {flag_emoji} |")
        
        return "\n".join(lines)
    
    @classmethod
    def _format_medications(cls, data: Dict[str, Any]) -> str:
        """Format medication list."""
        meds = data.get("medications", [])
        if not meds:
            return ""
        
        lines = ["## Current Medications", ""]
        for med in meds:
            name = med.get("name", "Unknown")
            dosage = med.get("dosage", "")
            freq = med.get("frequency", "")
            lines.append(f"- **{name}** {dosage} {freq}".strip())
        
        return "\n".join(lines)


# ============================================================================
# Specialty Templates
# ============================================================================

class CardiologyTemplate(ReportTemplate):
    """Template for cardiology reports."""
    
    name = "cardiology"
    display_name = "Cardiology Consultation"
    
    sections = [
        "summary",
        "cardiac_history",
        "vitals",
        "cardiac_labs",
        "ecg_findings",
        "imaging",
        "risk_stratification",
        "medications",
        "recommendations"
    ]
    
    @classmethod
    def _format_cardiac_labs(cls, data: Dict[str, Any]) -> str:
        """Format cardiac-specific labs."""
        labs = data.get("labs", [])
        cardiac_tests = ["troponin", "bnp", "nt-probnp", "ck-mb", "ldl", "hdl", "cholesterol"]
        
        cardiac_labs = [
            lab for lab in labs 
            if any(t in lab.get("test_name", "").lower() for t in cardiac_tests)
        ]
        
        if not cardiac_labs:
            return ""
        
        lines = ["## Cardiac Biomarkers", "", "| Marker | Value | Status |", "|--------|-------|--------|"]
        for lab in cardiac_labs:
            flag = lab.get("flag", "")
            status = "⚠️ Abnormal" if flag in ["H", "L", "C"] else "Normal"
            lines.append(f"| {lab.get('test_name')} | {lab.get('value')} {lab.get('unit', '')} | {status} |")
        
        return "\n".join(lines)
    
    @classmethod
    def _format_risk_stratification(cls, data: Dict[str, Any]) -> str:
        """Format cardiac risk assessment."""
        return """## Risk Stratification

Based on available data, cardiovascular risk factors include:
- *Assessment pending clinician review*

**ASCVD Risk Score**: *To be calculated*
"""


class NephrologyTemplate(ReportTemplate):
    """Template for nephrology reports."""
    
    name = "nephrology"
    display_name = "Nephrology Consultation"
    
    sections = [
        "summary",
        "renal_history",
        "renal_labs",
        "urinalysis",
        "fluid_status",
        "medications",
        "dialysis_assessment",
        "recommendations"
    ]
    
    @classmethod
    def _format_renal_labs(cls, data: Dict[str, Any]) -> str:
        """Format renal function labs."""
        labs = data.get("labs", [])
        renal_tests = ["creatinine", "bun", "egfr", "cystatin", "potassium", "phosphorus"]
        
        renal_labs = [
            lab for lab in labs 
            if any(t in lab.get("test_name", "").lower() for t in renal_tests)
        ]
        
        if not renal_labs:
            return ""
        
        lines = ["## Renal Function Panel", "", "| Test | Value | Reference | Stage |", "|------|-------|-----------|-------|"]
        
        for lab in renal_labs:
            # Determine CKD stage from eGFR
            stage = ""
            if "egfr" in lab.get("test_name", "").lower():
                try:
                    egfr = float(lab.get("value", 0))
                    if egfr >= 90:
                        stage = "G1"
                    elif egfr >= 60:
                        stage = "G2"
                    elif egfr >= 45:
                        stage = "G3a"
                    elif egfr >= 30:
                        stage = "G3b"
                    elif egfr >= 15:
                        stage = "G4"
                    else:
                        stage = "G5"
                except:
                    pass
            
            lines.append(f"| {lab.get('test_name')} | {lab.get('value')} {lab.get('unit', '')} | {lab.get('reference_range', '')} | {stage} |")
        
        return "\n".join(lines)


class OncologyTemplate(ReportTemplate):
    """Template for oncology reports."""
    
    name = "oncology"
    display_name = "Oncology Consultation"
    
    sections = [
        "summary",
        "cancer_history",
        "staging",
        "tumor_markers",
        "labs",
        "treatment_history",
        "current_regimen",
        "side_effects",
        "recommendations"
    ]
    
    @classmethod
    def _format_tumor_markers(cls, data: Dict[str, Any]) -> str:
        """Format tumor marker trends."""
        labs = data.get("labs", [])
        markers = ["psa", "cea", "ca-125", "ca-19-9", "afp", "hcg"]
        
        marker_labs = [
            lab for lab in labs 
            if any(m in lab.get("test_name", "").lower() for m in markers)
        ]
        
        if not marker_labs:
            return ""
        
        lines = ["## Tumor Markers", "", "| Marker | Value | Trend |", "|--------|-------|-------|"]
        for lab in marker_labs:
            trend = "→"  # Would compare with historical
            lines.append(f"| {lab.get('test_name')} | {lab.get('value')} {lab.get('unit', '')} | {trend} |")
        
        return "\n".join(lines)


class EndocrinologyTemplate(ReportTemplate):
    """Template for endocrinology reports."""
    
    name = "endocrinology"
    display_name = "Endocrinology Consultation"
    
    sections = [
        "summary",
        "diabetes_management",
        "thyroid_panel",
        "metabolic_labs",
        "medications",
        "lifestyle",
        "recommendations"
    ]
    
    @classmethod
    def _format_diabetes_management(cls, data: Dict[str, Any]) -> str:
        """Format diabetes-specific data."""
        labs = data.get("labs", [])
        
        # Find HbA1c
        hba1c = next((l for l in labs if "a1c" in l.get("test_name", "").lower()), None)
        glucose = next((l for l in labs if "glucose" in l.get("test_name", "").lower()), None)
        
        if not hba1c and not glucose:
            return ""
        
        lines = ["## Diabetes Management", ""]
        
        if hba1c:
            a1c_val = hba1c.get("value", "")
            try:
                a1c_num = float(a1c_val.replace("%", ""))
                if a1c_num < 7:
                    status = "✅ At goal"
                elif a1c_num < 8:
                    status = "⚠️ Above goal"
                else:
                    status = "🔴 Uncontrolled"
            except:
                status = ""
            lines.append(f"**HbA1c**: {a1c_val} {status}")
        
        if glucose:
            lines.append(f"**Fasting Glucose**: {glucose.get('value')} {glucose.get('unit', '')}")
        
        return "\n".join(lines)


# ============================================================================
# Template Registry
# ============================================================================

TEMPLATES = {
    "generic": ReportTemplate,
    "cardiology": CardiologyTemplate,
    "nephrology": NephrologyTemplate,
    "oncology": OncologyTemplate,
    "endocrinology": EndocrinologyTemplate,
}


def get_template(specialty: str = "generic") -> ReportTemplate:
    """Get report template by specialty."""
    return TEMPLATES.get(specialty.lower(), ReportTemplate)


def list_templates() -> List[Dict[str, str]]:
    """List available templates."""
    return [
        {"name": t.name, "display_name": t.display_name}
        for t in TEMPLATES.values()
    ]


def format_report(data: Dict[str, Any], specialty: str = "generic") -> str:
    """Format report using appropriate template."""
    template = get_template(specialty)
    return template.format_report(data)
