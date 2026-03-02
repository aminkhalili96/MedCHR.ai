"""
LLM Integration Module for MedCHR.ai
Handles text generation, embeddings, and clinical analysis using Mistral-compatible endpoints.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIError, AuthenticationError

from .config import get_settings
from .llm_gateway import ensure_phi_processor

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMTimeoutError(LLMError):
    """Exception for LLM timeout errors."""
    pass


class LLMContentError(LLMError):
    """Exception for LLM content policy violations."""
    pass


def get_client() -> OpenAI:
    """Get configured OpenAI client for Mistral endpoints."""
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or "https://api.openai.com/v1",
        max_retries=3,
        timeout=30.0,
    )
    return client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIError, ConnectionError)),
)
def generate_text(
    prompt: str,
    system_message: str = "You are a helpful healthcare assistant.",
    max_tokens: int = 500,
    temperature: float = 0.2,
    model: Optional[str] = None,
    json_mode: bool = False,
) -> str:
    """
    Generate text using the LLM.
    
    Args:
        prompt: The user prompt
        system_message: System instructions
        max_tokens: Maximum tokens to generate
        temperature: Creativity level (0-1)
        model: Specific model to use
        json_mode: Whether to force JSON output
    
    Returns:
        Generated text content
    """
    ensure_phi_processor("openai")
    
    model_name = model or settings.openai_model or "gpt-4o-mini"
    client = get_client()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
        )
        
        if not response.choices:
            raise LLMError("No response from LLM")
        
        content = response.choices[0].message.content
        if not content:
            raise LLMError("Empty response from LLM")
        
        return content.strip()
        
    except AuthenticationError as e:
        raise LLMError(f"LLM authentication failed: {str(e)}")
    except APIError as e:
        if "content_policy" in str(e).lower():
            raise LLMContentError(f"Content policy violation: {str(e)}")
        raise LLMError(f"LLM API error: {str(e)}")
    except Exception as e:
        raise LLMError(f"LLM generation failed: {str(e)}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIError, ConnectionError)),
)
def generate_embeddings(
    texts: List[str],
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model: Specific embedding model to use
    
    Returns:
        List of embedding vectors
    """
    ensure_phi_processor("openai")
    
    if not texts:
        return []
    
    model_name = model or settings.openai_embedding_model or "text-embedding-3-large"
    client = get_client()
    
    try:
        response = client.embeddings.create(
            model=model_name,
            input=texts,
        )
        
        if not response.data:
            raise LLMError("No embedding data returned")
        
        return [item.embedding for item in response.data]
        
    except AuthenticationError as e:
        raise LLMError(f"LLM authentication failed: {str(e)}")
    except APIError as e:
        raise LLMError(f"LLM embedding API error: {str(e)}")
    except Exception as e:
        raise LLMError(f"Embedding generation failed: {str(e)}")


def generate_structured_output(
    prompt: str,
    schema: Dict[str, Any],
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Generate structured JSON output using the LLM.
    
    Args:
        prompt: The user prompt
        schema: JSON schema for expected output format
        max_retries: Maximum retry attempts
    
    Returns:
        Parsed JSON dictionary
    """
    system_message = f"""You are a healthcare data structuring assistant.
    Respond ONLY with valid JSON matching this schema: {json.dumps(schema, indent=2)}
    Do not include any additional text, explanations, or markdown formatting."""
    
    for attempt in range(max_retries):
        try:
            result = generate_text(
                prompt=prompt,
                system_message=system_message,
                json_mode=True,
                max_tokens=2000,
            )
            return json.loads(result)
        except (json.JSONDecodeError, LLMContentError) as e:
            if attempt == max_retries - 1:
                raise LLMError(f"Failed to generate valid structured output: {str(e)}")
            time.sleep(1)
    
    raise LLMError("Max retries exceeded for structured output")


def generate_clinical_summary(
    patient_data: Dict[str, Any],
    focus_areas: Optional[List[str]] = None,
) -> str:
    """
    Generate a clinical summary from patient data.
    
    Args:
        patient_data: Structured patient data
        focus_areas: Specific areas to focus on
    
    Returns:
        Clinical summary text
    """
    focus_text = ""
    if focus_areas:
        focus_text = f" Focus specifically on: {', '.join(focus_areas)}"
    
    prompt = f"""Generate a concise clinical summary from this patient data:
    {json.dumps(patient_data, indent=2)}
    
    Include:
    - Key diagnoses and conditions
    - Current medications
    - Recent lab results and trends
    - Clinical recommendations
    - Any safety concerns or alerts
    
    {focus_text}
    
    Format as professional medical summary suitable for clinician review."""
    
    return generate_text(
        prompt=prompt,
        system_message="You are an experienced clinician generating professional medical summaries.",
        max_tokens=1500,
        temperature=0.1,
    )


def answer_clinical_question(
    question: str,
    context: str,
    patient_name: str = "Patient",
) -> str:
    """
    Answer a clinical question with proper citations.
    
    Args:
        question: The clinical question
        context: Relevant context from patient records
        patient_name: Patient name for personalization
    
    Returns:
        Answer with citations
    """
    prompt = f"""Answer this clinical question about {patient_name}:
    
    Question: {question}
    
    Context from patient records:
    {context}
    
    Provide a concise, evidence-based answer. Include citations to specific 
    parts of the context using format [1], [2], etc. List citations at the end.
    
    If the answer cannot be determined from the context, state that clearly."""
    
    return generate_text(
        prompt=prompt,
        system_message="You are a helpful clinical decision support assistant.",
        max_tokens=800,
        temperature=0.1,
    )


def extract_clinical_entities(
    text: str,
) -> Dict[str, List[str]]:
    """
    Extract clinical entities from text.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary of entity types and extracted entities
    """
    schema = {
        "diagnoses": "list[str]",
        "medications": "list[str]",
        "lab_tests": "list[str]",
        "procedures": "list[str]",
        "symptoms": "list[str]",
    }
    
    prompt = f"""Extract all clinical entities from this text:
    
    {text}
    
    Return ONLY a JSON object with these keys: {json.dumps(schema)}
    Each value should be a list of strings."""
    
    try:
        result = generate_structured_output(prompt, schema)
        return result
    except Exception as e:
        logger.warning(f"Clinical entity extraction failed: {str(e)}")
        return {
            "diagnoses": [],
            "medications": [],
            "lab_tests": [],
            "procedures": [],
            "symptoms": [],
        }


def generate_chr_report(
    patient_data: Dict[str, Any],
    clinician_notes: str = "",
) -> Dict[str, Any]:
    """
    Generate a comprehensive Client Health Report (CHR).
    
    Args:
        patient_data: Structured patient data
        clinician_notes: Additional notes from clinician
    
    Returns:
        Complete CHR report dictionary
    """
    schema = {
        "patient_summary": "str",
        "key_findings": "list[str]",
        "diagnoses": "list[dict]",
        "medications": "list[dict]",
        "lab_results": "list[dict]",
        "clinical_recommendations": "list[str]",
        "safety_alerts": "list[str]",
        "follow_up_plan": "str",
        "citations": "list[dict]",
    }
    
    prompt = f"""Generate a comprehensive Client Health Report (CHR) from this patient data:
    
    Patient Data:
    {json.dumps(patient_data, indent=2)}
    
    Clinician Notes:
    {clinician_notes}
    
    Generate a professional, structured CHR with all required sections.
    Include citations to source documents where appropriate.
    Format diagnoses, medications, and lab results as structured data.
    Highlight any safety concerns or clinical alerts."""
    
    return generate_structured_output(prompt, schema)


def generate_diagnosis_suggestions(
    patient_summary: str,
    symptoms: List[str],
    lab_results: List[Dict[str, Any]],
    medications: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate differential diagnosis suggestions.
    
    Args:
        patient_summary: Brief patient summary
        symptoms: List of current symptoms
        lab_results: Recent lab results
        medications: Current medications
    
    Returns:
        List of diagnosis suggestions with confidence scores
    """
    prompt = f"""Suggest differential diagnoses for this patient:
    
    Summary: {patient_summary}
    
    Symptoms: {', '.join(symptoms) if symptoms else 'None reported'}
    
    Lab Results:
    {json.dumps(lab_results, indent=2)}
    
    Current Medications: {', '.join(medications) if medications else 'None'}
    
    Provide 3-5 most likely diagnoses with:
    - Diagnosis name
    - ICD-10 code (if known)
    - Rationale based on presented data
    - Confidence level (High/Medium/Low)
    
    Return as JSON array."""
    
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "string"},
                "icd10": {"type": "string"},
                "rationale": {"type": "string"},
                "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            },
            "required": ["diagnosis", "rationale", "confidence"],
        },
    }
    
    try:
        result = generate_structured_output(prompt, schema)
        return result.get("items", [])
    except Exception as e:
        logger.error(f"Diagnosis suggestion generation failed: {str(e)}")
        return []


def analyze_lab_trends(
    lab_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze laboratory trends over time.
    
    Args:
        lab_data: List of lab results with dates
    
    Returns:
        Trend analysis with insights
    """
    prompt = f"""Analyze these laboratory trends:
    
    {json.dumps(lab_data, indent=2)}
    
    Identify:
    - Significant upward or downward trends
    - Values outside reference ranges
    - Potential clinical significance
    - Recommendations for follow-up
    
    Provide a structured analysis with clear insights."""
    
    schema = {
        "trends_identified": "list[str]",
        "abnormal_values": "list[dict]",
        "clinical_insights": "list[str]",
        "recommendations": "list[str]",
    }
    
    return generate_structured_output(prompt, schema)