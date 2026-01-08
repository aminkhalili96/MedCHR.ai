from typing import Dict, Any, List, Optional


def _parse_labs(text: str) -> List[Dict[str, Any]]:
    labs: List[Dict[str, Any]] = []
    panel: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("panel:"):
            panel = line.split(":", 1)[1].strip()
            continue
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        if parts[0].lower() in {"test", "patient", "panel"}:
            continue
        test = parts[0]
        value = parts[1] if len(parts) > 1 else ""
        unit = parts[2] if len(parts) > 2 else ""
        ref_range = parts[3] if len(parts) > 3 else ""
        flag = parts[4] if len(parts) > 4 else ""
        labs.append(
            {
                "panel": panel,
                "test": test,
                "value": value,
                "unit": unit,
                "range": ref_range,
                "flag": flag,
            }
        )
    return labs


def _parse_medications(text: str) -> List[str]:
    meds: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("medication") or lower.startswith("supplements"):
            continue
        if any(token in lower for token in [" mg", " mcg", " iu", " units"]) and any(
            char.isdigit() for char in line
        ):
            meds.append(line)
    return meds


def _parse_diagnoses(text: str) -> List[str]:
    diagnoses: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("assessment:") or lower.startswith("impression:"):
            payload = line.split(":", 1)[1].strip()
            for item in payload.split(","):
                value = item.strip()
                if value:
                    diagnoses.append(value)
    return diagnoses


def extract_structured(text: str) -> Dict[str, Any]:
    labs = _parse_labs(text)
    meds = _parse_medications(text)
    diagnoses = _parse_diagnoses(text)
    return {
        "labs": labs,
        "biomarkers": labs,
        "diagnoses": diagnoses,
        "medications": meds,
        "procedures": [],
        "genetics": [],
        "notes": text[:2000],
    }
