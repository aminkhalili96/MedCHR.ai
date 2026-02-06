"""
Lab Name Normalizer - Gap 3: Semantic Normalization
Maps diverse lab test names to canonical LOINC-based names.
"""
import json
import os
from pathlib import Path
from functools import lru_cache

DATA_PATH = Path(__file__).parent.parent / "data" / "loinc_subset.json"


@lru_cache(maxsize=1)
def _load_loinc_data() -> dict:
    """Load LOINC reference data."""
    if not DATA_PATH.exists():
        return {"tests": []}
    with open(DATA_PATH) as f:
        return json.load(f)


def _build_alias_map() -> dict[str, str]:
    """Build a mapping from all aliases to canonical names."""
    data = _load_loinc_data()
    alias_map = {}
    for test in data.get("tests", []):
        canonical = test["canonical"]
        # Map canonical to itself
        alias_map[canonical.lower()] = canonical
        # Map all aliases
        for alias in test.get("aliases", []):
            alias_map[alias.lower()] = canonical
    return alias_map


_ALIAS_MAP = None


def get_alias_map() -> dict[str, str]:
    """Get the alias map, building it if necessary."""
    global _ALIAS_MAP
    if _ALIAS_MAP is None:
        _ALIAS_MAP = _build_alias_map()
    return _ALIAS_MAP


def normalize_lab_name(raw_name: str) -> str:
    """
    Normalize a lab test name to its canonical form.
    
    Args:
        raw_name: The raw lab test name from the document (e.g., "WBC", "White Count")
        
    Returns:
        The canonical name (e.g., "WBC") or the original if no match found.
    """
    if not raw_name:
        return raw_name
    
    alias_map = get_alias_map()
    normalized = alias_map.get(raw_name.strip().lower())
    
    if normalized:
        return normalized
    
    # Try partial matching for common patterns
    raw_lower = raw_name.strip().lower()
    for alias, canonical in alias_map.items():
        if alias in raw_lower or raw_lower in alias:
            return canonical
    
    # Return original if no match
    return raw_name.strip()


def normalize_lab_list(labs: list[dict]) -> list[dict]:
    """
    Normalize a list of lab results.
    
    Args:
        labs: List of lab dictionaries with 'test' or 'test_name' keys
        
    Returns:
        List with normalized test names and original names preserved.
    """
    normalized = []
    for lab in labs:
        lab_copy = lab.copy()
        # Handle different key names
        test_key = "test" if "test" in lab else "test_name"
        original_name = lab.get(test_key, "")
        
        lab_copy["original_name"] = original_name
        lab_copy[test_key] = normalize_lab_name(original_name)
        lab_copy["canonical_name"] = lab_copy[test_key]
        
        normalized.append(lab_copy)
    
    return normalized


def get_loinc_code(canonical_name: str) -> str | None:
    """Get the LOINC code for a canonical lab name."""
    data = _load_loinc_data()
    for test in data.get("tests", []):
        if test["canonical"].lower() == canonical_name.lower():
            return test.get("loinc")
    return None
