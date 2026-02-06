from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PatientCreate(BaseModel):
    full_name: str = Field(..., min_length=1)
    dob: Optional[date] = None
    notes: Optional[str] = None


class Patient(BaseModel):
    id: str
    full_name: str
    dob: Optional[date] = None
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


class SignedUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: Optional[str] = None
    expires_in_seconds: int = Field(default=600, ge=60, le=3600)


class SignedUploadRegistration(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: Optional[str] = None
    storage_path: str = Field(..., min_length=3)


class SignedUploadResponse(BaseModel):
    patient_id: str
    filename: str
    content_type: str
    storage_path: str
    upload_url: str
    upload_token: str
    expires_in_seconds: int


class SignedDownloadResponse(BaseModel):
    document_id: str
    storage_path: str
    download_url: str
    expires_in_seconds: int


class LabResult(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    flag: Optional[str] = None  # High, Low, Normal
    reference_range: Optional[str] = None
    date: Optional[date] = None
    panel: Optional[str] = None


class Medication(BaseModel):
    name: str
    dosage: str
    frequency: Optional[str] = None
    route: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Optional[str] = "active"  # active, discontinued, completed


class Diagnosis(BaseModel):
    condition: str
    code: Optional[str] = None  # ICD-10 or SNOMED
    status: Optional[str] = None  # Active, Resolved, History
    date_onset: Optional[date] = None


class ExtractionData(BaseModel):
    labs: List[LabResult] = []
    medications: List[Medication] = []
    diagnoses: List[Diagnosis] = []
    notes: Optional[str] = None
    
    
class ExtractionResult(BaseModel):
    document_id: str
    raw_text: str
    structured: ExtractionData


class ChrDraftRequest(BaseModel):
    patient_id: str
    notes: Optional[str] = None


class ChrDraft(BaseModel):
    patient_id: str
    draft: Dict[str, Any]
    citations: List[Dict[str, Any]]


class Allergy(BaseModel):
    id: Optional[str] = None
    substance: str
    reaction: Optional[str] = None
    severity: Optional[str] = None
    status: str = "active"

class Vital(BaseModel):
    id: Optional[str] = None
    type: str
    value_1: float
    value_2: Optional[float] = None
    unit: Optional[str] = None
    recorded_at: datetime 

class Immunization(BaseModel):
    id: Optional[str] = None
    vaccine_name: str
    date_administered: Optional[date] = None
    status: str = "completed"

class JobStatus(BaseModel):
    job_id: str
    status: str

class EmbedResult(BaseModel):
    document_id: str
    chunks: int
