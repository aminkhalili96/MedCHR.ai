"""
Trend Analysis Module

Provides longitudinal health tracking and trend detection.

Gap Reference: C06
"""

from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    INSUFFICIENT_DATA = "insufficient_data"


class TrendSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


def calculate_trend(values: List[dict], test_name: str) -> dict:
    """
    Calculate trend for a lab value over time.
    
    Args:
        values: List of {value, date, unit} dicts, ordered by date
        test_name: Name of the test for reference range lookup
        
    Returns:
        Dict with trend direction, percent change, and alerts
    """
    if len(values) < 2:
        return {
            "direction": TrendDirection.INSUFFICIENT_DATA,
            "message": "Need at least 2 values for trend analysis"
        }
    
    # Sort by date
    sorted_values = sorted(values, key=lambda x: x.get("date", ""))
    
    # Get numeric values
    numeric_values = []
    for v in sorted_values:
        try:
            numeric_values.append({
                "value": float(str(v.get("value", "")).replace(",", "")),
                "date": v.get("date")
            })
        except ValueError:
            continue
    
    if len(numeric_values) < 2:
        return {
            "direction": TrendDirection.INSUFFICIENT_DATA,
            "message": "Not enough numeric values"
        }
    
    # Calculate percent change
    first = numeric_values[0]["value"]
    last = numeric_values[-1]["value"]
    
    if first == 0:
        pct_change = 0
    else:
        pct_change = ((last - first) / abs(first)) * 100
    
    # Calculate trend direction based on test type
    higher_is_worse = is_higher_worse(test_name)
    
    if abs(pct_change) < 5:
        direction = TrendDirection.STABLE
    elif higher_is_worse:
        direction = TrendDirection.WORSENING if pct_change > 0 else TrendDirection.IMPROVING
    else:
        direction = TrendDirection.IMPROVING if pct_change > 0 else TrendDirection.WORSENING
    
    # Calculate rate of change
    if numeric_values[0].get("date") and numeric_values[-1].get("date"):
        try:
            start_date = datetime.fromisoformat(numeric_values[0]["date"])
            end_date = datetime.fromisoformat(numeric_values[-1]["date"])
            days = (end_date - start_date).days
            rate_per_day = abs(last - first) / max(days, 1)
        except:
            days = None
            rate_per_day = None
    else:
        days = None
        rate_per_day = None
    
    result = {
        "direction": direction,
        "percent_change": round(pct_change, 1),
        "first_value": first,
        "last_value": last,
        "data_points": len(numeric_values)
    }
    
    if days:
        result["time_span_days"] = days
        result["rate_per_day"] = rate_per_day
    
    # Add alerts for concerning trends
    if direction == TrendDirection.WORSENING:
        if abs(pct_change) > 50:
            result["severity"] = TrendSeverity.CRITICAL
            result["alert"] = f"Rapid worsening: {test_name} changed by {round(pct_change, 1)}%"
        elif abs(pct_change) > 20:
            result["severity"] = TrendSeverity.WARNING
            result["alert"] = f"Concerning trend: {test_name} worsening"
    
    return result


def is_higher_worse(test_name: str) -> bool:
    """
    Determine if higher values are worse for a given test.
    """
    higher_is_worse_tests = {
        "creatinine", "bun", "glucose", "hba1c", "a1c",
        "triglycerides", "ldl", "cholesterol",
        "alt", "ast", "alp", "bilirubin",
        "psa", "bnp", "troponin",
        "wbc", "neutrophils",
        "potassium",  # Both extremes are bad
    }
    
    lower_is_worse_tests = {
        "egfr", "gfr",
        "hemoglobin", "hematocrit", "rbc",
        "platelets",
        "hdl",
        "sodium",  # Low is usually worse
        "albumin",
    }
    
    test_lower = test_name.lower()
    
    if any(t in test_lower for t in higher_is_worse_tests):
        return True
    if any(t in test_lower for t in lower_is_worse_tests):
        return False
    
    # Default: higher is worse (conservative)
    return True


def analyze_patient_trends(labs_history: List[dict]) -> dict:
    """
    Analyze all lab trends for a patient.
    
    Args:
        labs_history: List of lab results with dates
        
    Returns:
        Dict of test_name -> trend analysis
    """
    # Group by test name
    by_test = {}
    for lab in labs_history:
        test = lab.get("test_name", "").strip()
        if not test:
            continue
        if test not in by_test:
            by_test[test] = []
        by_test[test].append(lab)
    
    # Calculate trends
    trends = {}
    alerts = []
    
    for test_name, values in by_test.items():
        trend = calculate_trend(values, test_name)
        trends[test_name] = trend
        
        if trend.get("alert"):
            alerts.append({
                "test": test_name,
                "alert": trend["alert"],
                "severity": trend.get("severity", TrendSeverity.INFO)
            })
    
    return {
        "trends": trends,
        "alerts": alerts,
        "summary": {
            "improving": sum(1 for t in trends.values() if t.get("direction") == TrendDirection.IMPROVING),
            "stable": sum(1 for t in trends.values() if t.get("direction") == TrendDirection.STABLE),
            "worsening": sum(1 for t in trends.values() if t.get("direction") == TrendDirection.WORSENING),
            "critical_alerts": sum(1 for a in alerts if a.get("severity") == TrendSeverity.CRITICAL)
        }
    }


def generate_trend_chart_data(values: List[dict]) -> dict:
    """
    Generate data structure for trend visualization.
    """
    sorted_values = sorted(values, key=lambda x: x.get("date", ""))
    
    return {
        "labels": [v.get("date", "") for v in sorted_values],
        "data": [
            float(str(v.get("value", 0)).replace(",", "")) 
            for v in sorted_values 
            if v.get("value")
        ],
        "reference_range": {
            "min": values[0].get("reference_min") if values else None,
            "max": values[0].get("reference_max") if values else None
        }
    }
