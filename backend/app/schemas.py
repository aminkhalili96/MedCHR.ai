from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PatientCreate(BaseModel):
    full_name: str = Field(..., min_length=1)
    dob: Optional[str] = None
    notes: Optional[str] = None


class Patient(BaseModel):
    id: str
    full_name: str
    dob: Optional[str] = None
    notes: Optional[str] = None
    lifestyle: Optional[Dict[str, Any]] = None
    genetics: Optional[Dict[str, Any]] = None


class DocumentCreate(BaseModel):
    patient_id: str
    filename: str
    content_type: str


class Document(BaseModel):
    id: str
    patient_id: str
    filename: str
    content_type: str
    storage_path: str


class ExtractionResult(BaseModel):
    document_id: str
    raw_text: str
    structured: Dict[str, Any]


class ChrDraftRequest(BaseModel):
    patient_id: str
    notes: Optional[str] = None


class ChrDraft(BaseModel):
    patient_id: str
    draft: Dict[str, Any]
    citations: List[Dict[str, Any]]
