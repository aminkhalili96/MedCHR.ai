import pytest
from fastapi import HTTPException

from backend.app.main import _validate_storage_path_for_patient


def test_validate_storage_path_for_patient_allows_patient_prefix():
    patient_id = "00000000-0000-0000-0000-000000000123"
    path = f"/{patient_id}/abc_report.pdf"
    assert _validate_storage_path_for_patient(patient_id, path) == f"{patient_id}/abc_report.pdf"


def test_validate_storage_path_for_patient_rejects_other_patient():
    with pytest.raises(HTTPException):
        _validate_storage_path_for_patient(
            "00000000-0000-0000-0000-000000000123",
            "00000000-0000-0000-0000-000000000456/abc_report.pdf",
        )


def test_validate_storage_path_for_patient_rejects_traversal():
    with pytest.raises(HTTPException):
        _validate_storage_path_for_patient(
            "00000000-0000-0000-0000-000000000123",
            "00000000-0000-0000-0000-000000000123/../../etc/passwd",
        )
