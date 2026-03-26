"""Tests for the clinical alerts engine — critical values, drug interactions, allergies."""

from backend.app.alerts import (
    check_critical_values,
    check_drug_interactions,
    check_allergy_contraindications,
    check_abnormal_trend,
    run_safety_checks,
    parse_numeric,
)


# ── parse_numeric ──────────────────────────────────────────────

class TestParseNumeric:
    def test_int(self):
        assert parse_numeric(42) == 42.0

    def test_float_str(self):
        assert parse_numeric("7.35") == 7.35

    def test_with_comparator(self):
        assert parse_numeric("<0.04") == 0.04

    def test_range_takes_average(self):
        assert parse_numeric("120-140") == 130.0

    def test_none_returns_none(self):
        assert parse_numeric(None) is None

    def test_non_numeric_returns_none(self):
        assert parse_numeric("normal") is None

    def test_comma_thousands(self):
        assert parse_numeric("1,200") == 1200.0


# ── check_critical_values ──────────────────────────────────────

class TestCriticalValues:
    def test_critically_high_potassium(self):
        labs = [{"test_name": "Potassium", "value": "7.0", "unit": "mEq/L"}]
        alerts = check_critical_values(labs)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "CRITICAL"
        assert alerts[0]["direction"] == "CRITICALLY HIGH"

    def test_critically_low_glucose(self):
        labs = [{"test_name": "Glucose", "value": "30", "unit": "mg/dL"}]
        alerts = check_critical_values(labs)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "CRITICAL"
        assert alerts[0]["direction"] == "CRITICALLY LOW"

    def test_normal_value_no_alert(self):
        labs = [{"test_name": "Potassium", "value": "4.0", "unit": "mEq/L"}]
        alerts = check_critical_values(labs)
        assert len(alerts) == 0

    def test_abnormal_but_not_critical_generates_warning(self):
        # HbA1c > 6.5 is abnormal but not in CRITICAL_RANGES
        labs = [{"test_name": "HbA1c", "value": "8.5", "unit": "%"}]
        alerts = check_critical_values(labs)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "WARNING"

    def test_non_numeric_value_skipped(self):
        labs = [{"test_name": "Potassium", "value": "pending", "unit": "mEq/L"}]
        alerts = check_critical_values(labs)
        assert len(alerts) == 0

    def test_multiple_labs(self):
        labs = [
            {"test_name": "Potassium", "value": "7.0", "unit": "mEq/L"},
            {"test_name": "Sodium", "value": "140", "unit": "mEq/L"},
            {"test_name": "Glucose", "value": "30", "unit": "mg/dL"},
        ]
        alerts = check_critical_values(labs)
        critical = [a for a in alerts if a["severity"] == "CRITICAL"]
        assert len(critical) == 2  # potassium + glucose


# ── check_drug_interactions ────────────────────────────────────

class TestDrugInteractions:
    def test_warfarin_aspirin(self):
        meds = ["Warfarin 5mg", "Aspirin 81mg"]
        alerts = check_drug_interactions(meds)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "HIGH"

    def test_ssri_maoi_critical(self):
        meds = ["Fluoxetine (SSRI)", "Phenelzine (MAOI)"]
        alerts = check_drug_interactions(meds)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "CRITICAL"

    def test_no_interaction(self):
        meds = ["Metformin 500mg", "Lisinopril 10mg"]
        alerts = check_drug_interactions(meds)
        assert len(alerts) == 0

    def test_empty_list(self):
        assert check_drug_interactions([]) == []

    def test_single_med_no_interaction(self):
        assert check_drug_interactions(["Warfarin"]) == []


# ── check_allergy_contraindications ────────────────────────────

class TestAllergyContraindications:
    def test_penicillin_allergy_amoxicillin(self):
        alerts = check_allergy_contraindications(["penicillin"], ["amoxicillin 500mg"])
        assert len(alerts) >= 1
        severities = {a["severity"] for a in alerts}
        assert "HIGH" in severities or "CRITICAL" in severities

    def test_no_allergy_match(self):
        alerts = check_allergy_contraindications(["shellfish"], ["metformin"])
        assert len(alerts) == 0

    def test_empty_inputs(self):
        assert check_allergy_contraindications([], []) == []


# ── check_abnormal_trend ──────────────────────────────────────

class TestAbnormalTrend:
    def test_significant_increase(self):
        current = [{"test_name": "Creatinine", "value": "3.0"}]
        historical = [{"test_name": "Creatinine", "value": "1.5"}]
        alerts = check_abnormal_trend(current, historical, threshold_pct=50)
        assert len(alerts) == 1
        assert alerts[0]["direction"] == "INCREASING"
        assert alerts[0]["percent_change"] == 100.0

    def test_no_significant_change(self):
        current = [{"test_name": "Glucose", "value": "105"}]
        historical = [{"test_name": "Glucose", "value": "100"}]
        alerts = check_abnormal_trend(current, historical, threshold_pct=50)
        assert len(alerts) == 0

    def test_significant_decrease(self):
        current = [{"test_name": "Hemoglobin", "value": "5.0"}]
        historical = [{"test_name": "Hemoglobin", "value": "12.0"}]
        alerts = check_abnormal_trend(current, historical, threshold_pct=50)
        assert len(alerts) == 1
        assert alerts[0]["direction"] == "DECREASING"


# ── run_safety_checks (integration) ───────────────────────────

class TestRunSafetyChecks:
    def test_consolidated_results(self):
        labs = [{"test_name": "Potassium", "value": "7.0", "unit": "mEq/L"}]
        meds = ["Warfarin", "Aspirin"]
        allergies = ["penicillin"]
        results = run_safety_checks(labs, meds, allergies)

        assert "critical_values" in results
        assert "drug_interactions" in results
        assert "allergy_alerts" in results
        assert "summary" in results
        assert results["summary"]["requires_immediate_action"] is True

    def test_clean_results(self):
        labs = [{"test_name": "Glucose", "value": "90", "unit": "mg/dL"}]
        results = run_safety_checks(labs, [], [])
        assert results["summary"]["critical_count"] == 0
        assert results["summary"]["requires_immediate_action"] is False
