"""
Timeline Generator - Gap 7: Patient Timeline View
Extracts and manages patient events for timeline visualization.
"""
from datetime import datetime
from typing import Optional
import re


def extract_date_from_text(text: str) -> Optional[str]:
    """
    Extract a date from text using common patterns.
    
    Returns:
        ISO date string (YYYY-MM-DD) or None if not found.
    """
    if not text:
        return None
    
    # Common date patterns
    patterns = [
        # 2025-01-10, 2025/01/10
        r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
        # 01/10/2025, 01-10-2025
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
        # January 10, 2025 or Jan 10, 2025
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})',
        # 10 January 2025
        r'(\d{1,2})\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            try:
                # Handle YYYY-MM-DD format
                if len(groups) == 3 and len(groups[0]) == 4:
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    return f"{year:04d}-{month:02d}-{day:02d}"
                # Handle MM-DD-YYYY format
                elif len(groups) == 3 and len(groups[2]) == 4:
                    month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except (ValueError, IndexError):
                continue
    
    return None


def extract_events_from_document(
    document_id: str,
    filename: str,
    raw_text: str,
    structured_data: dict
) -> list[dict]:
    """
    Extract timeline events from a processed document.
    
    Args:
        document_id: The document UUID
        filename: Original filename
        raw_text: Raw text content
        structured_data: Extracted structured data
        
    Returns:
        List of event dictionaries.
    """
    events = []
    
    # Try to extract service date
    service_date = extract_date_from_text(raw_text[:2000])  # Check first 2000 chars
    
    if not service_date:
        # Use today if no date found
        service_date = datetime.now().strftime("%Y-%m-%d")
    
    # Determine event type from filename or content
    filename_lower = filename.lower()
    
    if any(x in filename_lower for x in ["lab", "panel", "blood", "chem", "cbc", "cmp"]):
        event_type = "lab"
        title = "Laboratory Results"
    elif any(x in filename_lower for x in ["consult", "visit", "clinic", "office"]):
        event_type = "visit"
        title = "Clinical Visit"
    elif any(x in filename_lower for x in ["discharge", "admission", "hospital", "inpatient"]):
        event_type = "hospitalization"
        title = "Hospitalization"
    elif any(x in filename_lower for x in ["surgery", "procedure", "operative", "biopsy"]):
        event_type = "procedure"
        title = "Procedure"
    elif any(x in filename_lower for x in ["pathology", "genetic", "molecular"]):
        event_type = "diagnosis"
        title = "Pathology/Genetics Report"
    else:
        event_type = "document"
        title = "Medical Document"
    
    # Create main event for this document
    events.append({
        "event_type": event_type,
        "event_date": service_date,
        "title": title,
        "description": f"Processed: {filename}",
        "source_document_id": document_id
    })
    
    # Extract additional events from diagnoses
    diagnoses = structured_data.get("diagnoses", [])
    for dx in diagnoses[:5]:  # Limit
        if isinstance(dx, dict):
            dx_text = dx.get("condition", str(dx))
        else:
            dx_text = str(dx)
        
        events.append({
            "event_type": "diagnosis",
            "event_date": service_date,
            "title": f"Diagnosis: {dx_text[:50]}",
            "description": dx_text,
            "source_document_id": document_id
        })
    
    return events


def format_timeline_for_display(events: list[dict]) -> list[dict]:
    """
    Format events for frontend timeline display.
    
    Args:
        events: List of raw events from database
        
    Returns:
        Sorted, formatted events for display.
    """
    # Sort by date descending (most recent first)
    sorted_events = sorted(
        events,
        key=lambda x: x.get("event_date", "1900-01-01"),
        reverse=True
    )
    
    # Add display properties
    type_icons = {
        "lab": "🧪",
        "visit": "🏥",
        "hospitalization": "🛏️",
        "procedure": "⚕️",
        "diagnosis": "📋",
        "document": "📄"
    }
    
    type_colors = {
        "lab": "#3b82f6",      # blue
        "visit": "#10b981",    # green
        "hospitalization": "#ef4444",  # red
        "procedure": "#f59e0b",  # amber
        "diagnosis": "#8b5cf6",  # purple
        "document": "#6b7280"   # gray
    }
    
    for event in sorted_events:
        event_type = event.get("event_type", "document")
        event["icon"] = type_icons.get(event_type, "📄")
        event["color"] = type_colors.get(event_type, "#6b7280")
        
        # Format date for display
        date_str = event.get("event_date", "")
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                event["display_date"] = dt.strftime("%b %d, %Y")
            except ValueError:
                event["display_date"] = date_str
        else:
            event["display_date"] = "Unknown Date"
    
    return sorted_events
