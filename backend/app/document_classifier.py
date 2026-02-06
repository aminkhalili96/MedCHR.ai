"""
Document Classification Module

Automatically classifies medical documents by type:
- Lab Reports
- Consultation Notes
- Imaging Reports
- Discharge Summaries
- Pathology Reports
- Progress Notes

Gap Reference: D01
"""

import re
from typing import Optional
from enum import Enum


class DocumentType(str, Enum):
    LAB_REPORT = "lab_report"
    CONSULTATION = "consultation"
    IMAGING = "imaging"
    DISCHARGE = "discharge_summary"
    PATHOLOGY = "pathology"
    PROGRESS_NOTE = "progress_note"
    OPERATIVE = "operative_report"
    PRESCRIPTION = "prescription"
    REFERRAL = "referral"
    UNKNOWN = "unknown"


# Keywords for document classification
DOCUMENT_PATTERNS = {
    DocumentType.LAB_REPORT: [
        r"lab(?:oratory)?\s+report",
        r"test\s+results?",
        r"blood\s+work",
        r"urinalysis",
        r"reference\s+range",
        r"specimen",
        r"panel\s*:",
        r"chemistry",
        r"hematology",
        r"microbiology",
    ],
    DocumentType.CONSULTATION: [
        r"consultation\s+(?:note|report)",
        r"consult\s+(?:note|report)",
        r"referral\s+note",
        r"specialist\s+(?:note|report)",
        r"history\s+of\s+present\s+illness",
        r"chief\s+complaint",
        r"assessment\s*(?:and|/)\s*plan",
    ],
    DocumentType.IMAGING: [
        r"radiology\s+report",
        r"imaging\s+report",
        r"x-?ray",
        r"ct\s+scan",
        r"mri",
        r"ultrasound",
        r"mammogram",
        r"impression\s*:",
        r"findings\s*:",
        r"radiologist",
    ],
    DocumentType.DISCHARGE: [
        r"discharge\s+summary",
        r"discharge\s+instructions",
        r"hospital\s+course",
        r"discharge\s+diagnosis",
        r"discharge\s+medications",
        r"follow\s*-?\s*up\s+instructions",
    ],
    DocumentType.PATHOLOGY: [
        r"pathology\s+report",
        r"biopsy\s+report",
        r"cytology",
        r"histology",
        r"gross\s+description",
        r"microscopic\s+description",
        r"final\s+diagnosis",
        r"specimen\s+(?:type|received)",
    ],
    DocumentType.PROGRESS_NOTE: [
        r"progress\s+note",
        r"clinic\s+note",
        r"office\s+visit",
        r"follow\s*-?\s*up\s+(?:note|visit)",
        r"soap\s+note",
        r"encounter\s+note",
    ],
    DocumentType.OPERATIVE: [
        r"operative\s+(?:note|report)",
        r"procedure\s+(?:note|report)",
        r"surgical\s+(?:note|report)",
        r"anesthesia\s+(?:note|report)",
        r"pre-?operative",
        r"post-?operative",
    ],
    DocumentType.PRESCRIPTION: [
        r"prescription",
        r"rx\s*:",
        r"medication\s+order",
        r"sig\s*:",
        r"dispense",
        r"refill",
    ],
    DocumentType.REFERRAL: [
        r"referral",
        r"refer\s+to",
        r"please\s+(?:see|evaluate)",
        r"consultation\s+requested",
    ],
}


def classify_document(text: str, filename: str = "") -> dict:
    """
    Classify a document based on its content and filename.
    
    Args:
        text: Document text content
        filename: Optional filename for additional hints
        
    Returns:
        Dict with document_type, confidence, and matched_patterns
    """
    if not text:
        return {
            "document_type": DocumentType.UNKNOWN,
            "confidence": 0.0,
            "matched_patterns": []
        }
    
    text_lower = text.lower()
    filename_lower = filename.lower() if filename else ""
    
    scores = {}
    matches = {}
    
    for doc_type, patterns in DOCUMENT_PATTERNS.items():
        score = 0
        matched = []
        
        for pattern in patterns:
            # Check text content
            text_matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if text_matches:
                score += len(text_matches)
                matched.append(pattern)
            
            # Check filename (higher weight)
            if filename_lower and re.search(pattern, filename_lower, re.IGNORECASE):
                score += 3
                matched.append(f"filename:{pattern}")
        
        scores[doc_type] = score
        matches[doc_type] = matched
    
    # Find best match
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    
    if best_score == 0:
        return {
            "document_type": DocumentType.UNKNOWN,
            "confidence": 0.0,
            "matched_patterns": []
        }
    
    # Calculate confidence (normalized)
    max_possible = sum(len(patterns) for patterns in DOCUMENT_PATTERNS.values())
    confidence = min(best_score / 10, 1.0)  # Cap at 1.0
    
    return {
        "document_type": best_type,
        "confidence": round(confidence, 2),
        "matched_patterns": matches[best_type],
        "all_scores": {k.value: v for k, v in scores.items() if v > 0}
    }


def extract_document_date(text: str) -> Optional[str]:
    """
    Extract the primary date from a document.
    
    Gap Reference: D02
    """
    date_patterns = [
        # ISO format
        r"(\d{4}-\d{2}-\d{2})",
        # US format
        r"(\d{1,2}/\d{1,2}/\d{4})",
        # Written format
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
        # Report date label
        r"(?:date|dated|report\s+date|collection\s+date)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_provider_info(text: str) -> dict:
    """
    Extract provider/facility information from document.
    
    Gap Reference: D03, D04
    """
    result = {
        "provider_name": None,
        "facility_name": None,
        "provider_npi": None,
    }
    
    # Provider patterns
    provider_patterns = [
        r"(?:physician|doctor|provider|attending|ordered\s+by)\s*:?\s*([A-Z][a-zA-Z\s,\.]+(?:MD|DO|NP|PA))",
        r"([A-Z][a-z]+\s+[A-Z][a-z]+),?\s*(?:MD|DO|NP|PA)",
    ]
    
    for pattern in provider_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["provider_name"] = match.group(1).strip()
            break
    
    # Facility patterns
    facility_patterns = [
        r"(?:hospital|medical\s+center|clinic|laboratory)\s*:?\s*([A-Z][A-Za-z\s]+(?:Hospital|Medical|Center|Clinic|Lab))",
        r"((?:[A-Z][a-z]+\s+)+(?:Hospital|Medical Center|Clinic|Laboratory))",
    ]
    
    for pattern in facility_patterns:
        match = re.search(pattern, text)
        if match:
            result["facility_name"] = match.group(1).strip()
            break
    
    # NPI pattern
    npi_match = re.search(r"NPI\s*:?\s*(\d{10})", text)
    if npi_match:
        result["provider_npi"] = npi_match.group(1)
    
    return result


def calculate_document_quality(text: str, ocr_confidence: float = None) -> dict:
    """
    Calculate document quality score.
    
    Gap Reference: D08
    """
    issues = []
    
    # Check text length
    if len(text) < 100:
        issues.append("Very short document")
    
    # Check for OCR artifacts
    ocr_artifacts = re.findall(r"[^\x00-\x7F]{3,}", text)
    if len(ocr_artifacts) > 5:
        issues.append("Possible OCR errors detected")
    
    # Check for excessive special characters
    special_ratio = len(re.findall(r"[^a-zA-Z0-9\s\.,;:\-]", text)) / max(len(text), 1)
    if special_ratio > 0.1:
        issues.append("High ratio of special characters")
    
    # Check for readable words
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    if len(words) < 10:
        issues.append("Very few readable words")
    
    # Calculate score
    base_score = 1.0
    base_score -= len(issues) * 0.2
    
    if ocr_confidence is not None:
        base_score = (base_score + ocr_confidence) / 2
    
    quality_score = max(0, min(1, base_score))
    
    return {
        "quality_score": round(quality_score, 2),
        "issues": issues,
        "word_count": len(words),
        "is_acceptable": quality_score >= 0.5
    }
