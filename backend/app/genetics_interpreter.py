"""
Genetics Interpreter - Gap 4: Genetics Clinical Interpretation
Interprets genetic variants using PharmGKB reference data.
"""
import json
from pathlib import Path
from functools import lru_cache

DATA_PATH = Path(__file__).parent.parent / "data" / "pharmgkb_subset.json"


@lru_cache(maxsize=1)
def _load_pharmgkb_data() -> dict:
    """Load PharmGKB reference data."""
    if not DATA_PATH.exists():
        return {"genes": []}
    with open(DATA_PATH) as f:
        return json.load(f)


def interpret_variant(gene: str, variant: str) -> dict | None:
    """
    Interpret a genetic variant using PharmGKB data.
    
    Args:
        gene: The gene name (e.g., "CYP2D6", "MTHFR")
        variant: The variant (e.g., "*4/*4", "C677T Homozygous")
        
    Returns:
        Dictionary with phenotype, drugs_affected, and recommendation, or None if not found.
    """
    data = _load_pharmgkb_data()
    
    for gene_data in data.get("genes", []):
        if gene_data["gene"].upper() == gene.upper():
            variants = gene_data.get("variants", {})
            
            # Try exact match first
            if variant in variants:
                return {
                    "gene": gene,
                    "variant": variant,
                    **variants[variant]
                }
            
            # Try partial match (for variants like "C677T" matching "C677T Homozygous")
            for v_key, v_data in variants.items():
                if variant.lower() in v_key.lower() or v_key.lower() in variant.lower():
                    return {
                        "gene": gene,
                        "variant": v_key,
                        **v_data
                    }
    
    return None


def interpret_patient_genetics(genetics_data: dict) -> list[dict]:
    """
    Interpret all genetic findings for a patient.
    
    Args:
        genetics_data: Patient's genetics data (e.g., {"findings": [{"gene": "MTHFR", "variant": "C677T"}]})
        
    Returns:
        List of interpretations with clinical implications.
    """
    interpretations = []
    
    findings = genetics_data.get("findings", []) if genetics_data else []
    
    for finding in findings:
        gene = finding.get("gene", "")
        variant = finding.get("variant", "")
        
        interpretation = interpret_variant(gene, variant)
        
        if interpretation:
            interpretations.append(interpretation)
        else:
            # Return basic info even if no specific interpretation
            interpretations.append({
                "gene": gene,
                "variant": variant,
                "phenotype": finding.get("impact", "Unknown"),
                "drugs_affected": [],
                "recommendation": "No specific clinical guidance available in database. Consult pharmacogenomics resources."
            })
    
    return interpretations


def check_drug_gene_interactions(genetics_data: dict, medications: list[str]) -> list[dict]:
    """
    Check for drug-gene interactions given patient's genetics and current medications.
    
    Args:
        genetics_data: Patient's genetics data
        medications: List of current medication names
        
    Returns:
        List of interaction alerts.
    """
    alerts = []
    interpretations = interpret_patient_genetics(genetics_data)
    
    # Normalize medication names for matching
    med_names_lower = [m.lower() for m in medications]
    
    for interp in interpretations:
        drugs_affected = interp.get("drugs_affected", [])
        
        for drug in drugs_affected:
            if drug.lower() in med_names_lower or any(drug.lower() in m for m in med_names_lower):
                alerts.append({
                    "severity": "high",
                    "gene": interp["gene"],
                    "variant": interp["variant"],
                    "drug": drug,
                    "phenotype": interp.get("phenotype", ""),
                    "recommendation": interp.get("recommendation", ""),
                    "message": f"ALERT: {drug} interaction with {interp['gene']} {interp['variant']} ({interp.get('phenotype', '')})"
                })
    
    return alerts


def format_genetics_for_chr(genetics_data: dict) -> str:
    """
    Format genetics interpretations for inclusion in CHR.
    
    Returns:
        Markdown-formatted string for the CHR.
    """
    interpretations = interpret_patient_genetics(genetics_data)
    
    if not interpretations:
        return ""
    
    lines = ["## Pharmacogenomics Interpretation", ""]
    
    for interp in interpretations:
        lines.append(f"### {interp['gene']} - {interp['variant']}")
        lines.append(f"**Phenotype:** {interp.get('phenotype', 'Unknown')}")
        
        drugs = interp.get("drugs_affected", [])
        if drugs:
            lines.append(f"**Drugs Affected:** {', '.join(drugs)}")
        
        rec = interp.get("recommendation", "")
        if rec:
            lines.append(f"**Clinical Recommendation:** {rec}")
        
        lines.append("")
    
    return "\n".join(lines)
