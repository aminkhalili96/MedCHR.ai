"""
Diagnosis Suggester - Gap 5: Diagnosis Drafting/Suggestion
Generates AI-assisted preliminary diagnosis suggestions.
"""
import json
import logging

from .config import get_settings
from .llm_gateway import create_chat_completion, redact_if_enabled

logger = logging.getLogger(__name__)


ASSESSMENT_PROMPT = """You are a clinical decision support system helping a licensed physician.

Based on the following patient data, suggest potential NEW diagnoses or assessments to consider. 
These are suggestions for the physician to review, not definitive diagnoses.

Focus on:
1. Patterns in lab abnormalities
2. Combinations of findings that suggest specific conditions
3. Diagnoses that may explain multiple abnormalities

Patient Data:
{patient_summary}

Labs:
{labs_summary}

Current Medications:
{medications_summary}

Genetics:
{genetics_summary}

Existing Diagnoses:
{existing_diagnoses}

Provide 3-5 potential new diagnoses to consider. For each:
- State the diagnosis
- Provide a brief rationale based on the data
- Suggest an ICD-10 code if known
- Rate confidence as HIGH, MEDIUM, or LOW

Format as JSON array:
[
  {{"diagnosis": "...", "rationale": "...", "icd10": "...", "confidence": "..."}},
  ...
]

Only output the JSON array, no other text.
"""


def generate_diagnosis_suggestions(
    patient_summary: str,
    labs: list[dict],
    medications: list[str],
    genetics: dict,
    existing_diagnoses: list[str]
) -> list[dict]:
    """
    Generate AI-assisted diagnosis suggestions.
    
    Args:
        patient_summary: Brief patient description
        labs: List of lab results
        medications: List of current medications
        genetics: Patient's genetics data
        existing_diagnoses: List of already-known diagnoses
        
    Returns:
        List of suggested diagnoses with rationale.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return []

    # Format inputs
    labs_text = "\n".join([
        f"- {lab.get('test', lab.get('test_name', 'Unknown'))}: {lab.get('value', '')} {lab.get('unit', '')} ({lab.get('flag', 'Normal')})"
        for lab in labs[:20]  # Limit to avoid token overflow
    ]) or "No labs available"
    
    meds_text = ", ".join(medications[:15]) if medications else "No medications listed"
    
    genetics_text = ""
    if genetics and genetics.get("findings"):
        genetics_text = "\n".join([
            f"- {f.get('gene', '')}: {f.get('variant', '')} - {f.get('impact', '')}"
            for f in genetics.get("findings", [])
        ])
    else:
        genetics_text = "No genetic data available"
    
    existing_text = ", ".join(existing_diagnoses) if existing_diagnoses else "None documented"
    
    prompt = ASSESSMENT_PROMPT.format(
        patient_summary=patient_summary or "No summary available",
        labs_summary=labs_text,
        medications_summary=meds_text,
        genetics_summary=genetics_text,
        existing_diagnoses=existing_text
    )
    prompt = redact_if_enabled(prompt)
    
    try:
        response = create_chat_completion(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        suggestions = json.loads(content)
        return suggestions
        
    except Exception as e:
        logger.warning("Error generating diagnosis suggestions", exc_info=e)
        return []


def format_suggestions_for_chr(suggestions: list[dict]) -> str:
    """
    Format diagnosis suggestions for CHR display.
    
    Returns:
        Markdown-formatted string.
    """
    if not suggestions:
        return ""
    
    lines = ["## Suggested Assessments to Consider", ""]
    lines.append("*The following are AI-generated suggestions for physician review. They are not diagnoses.*")
    lines.append("")
    
    for i, sug in enumerate(suggestions, 1):
        confidence = sug.get("confidence", "MEDIUM")
        badge = "🔴" if confidence == "HIGH" else ("🟡" if confidence == "MEDIUM" else "🟢")
        
        lines.append(f"### {i}. {sug.get('diagnosis', 'Unknown')} {badge}")
        lines.append(f"**ICD-10:** {sug.get('icd10', 'Not specified')}")
        lines.append(f"**Rationale:** {sug.get('rationale', '')}")
        lines.append(f"**Confidence:** {confidence}")
        lines.append("")
    
    return "\n".join(lines)
