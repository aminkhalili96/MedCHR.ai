
import sys
import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

# Setup path to import backend modules
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Use the virtualenv python if running directly, but here we assume the runner 
# is already using the correct python environment or we just import.
try:
    from backend.app.db import get_conn
    from backend.app.config import get_settings
    from backend.app.embeddings import embed_texts
    from backend.app.storage import upload_bytes
except ImportError:
    # If running from incorrect context, this might fail unless paths are strict
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def chunk_text(text: str, size=1000, overlap=100):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    chunk_index = 0
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_text": chunk,
                "chunk_index": chunk_index,
                "chunk_start": start,
                "chunk_end": end
            })
            chunk_index += 1
        start = max(start + size - overlap, end) if end < length else length
    return chunks

# --- Data Definitions (14 Complex Patients) ---

# --- Helper for Generating Massive History ---
def generate_complex_history(patient_name, condition, extra_details=""):
    """
    Generates a long, complex, realistic medical history string.
    This simulates a 20-50 page medical record condensed into text.
    """
    base = f"""
Medical Record Summary for: {patient_name}
Primary Condition: {condition}
Generated: {datetime.now().strftime('%Y-%m-%d')}

--- EXECUTIVE SUMMARY ---
Patient is a complex case with a multi-year history of {condition}. 
Management has been complicated by {extra_details if extra_details else 'multiple comorbidities and medication adjustments'}.
Current status involves active disease management requiring frequent specialist interventions.

--- LONGITUDINAL HISTORY (2018 - Present) ---
[2018]: Initial presentation with non-specific symptoms. Primary care workup inconclusive.
    - Jan 12: Labs showed mild abnormalities. Referred to specialist.
    - Mar 04: First specialist consult. Diagnostic imaging ordered.
    - Apr 20: Diagnosis confirmed. Initial treatment regimen (Line 1) started.
    - Aug 15: 3-month follow-up. Partial response. Dosage increased.

[2019]: Disease progression noted.
    - Feb 10: ER Visit for acute exacerbation. Treated with IV steroids/antibiotics.
    - Feb 12: Discharged with tapered dose.
    - May 22: Routine follow-up. Stability achieved.
    - Nov 08: Patient reported new side effects (nausea, fatigue). Meds adjusted.

[2020]: Pandemic-era management (Telehealth).
    - Apr 05: Tele-visit. Compliance good. Labs ordered remotely.
    - Sep 12: Flare-up managed outpatient.
    - Dec 20: Annual review. Imaging shows slight progression. Discussion of biologic therapy.

[2021-2022]: Biologic Therapy Initiation.
    - Jan 15, 2021: Started Biologic Agent A.
    - Mar 01: marked improvement in symptoms.
    - Jun 10: Insurance denial for renewal. Gap in therapy of 3 weeks.
    - Jul 05: Appeal granted. Therapy resumed.
    - Oct 12, 2022: Breakthrough symptoms. Switch to Biologic Agent B considered.

[2023-2024]: Recent Course.
    - High utilization of healthcare resources.
    - Multiple consults: Cardiology, Pulmonology, Nephrology, Nutrition.
    - Recent hospitalization (Oct 2023) for pneumonia complications.
    - Hospital Course: admitted to floor, required O2 2L. IV Ceftriaxone/Azithro. Discharged day 5.

--- COMPREHENSIVE REVIEW OF SYSTEMS ---
General: Fatigue (chronic), weight fluctuation (+/- 5lbs).
HEENT: Occasional dry eyes.
CV: HTN controlled on meds. Occasional palpitations.
Pulm: Dyspnea on exertion (baseline).
GI: GERD symptoms, controlled with PPI.
GU: No dysuria.
MSK: Joint pains (knees, hands) worse in morning.
Neuro: Mild neuropathy in feet (likely diabetic/idiopathic).
Psych: Anxiety related to chronic illness.

--- SOCIAL HISTORY & DETERMINANTS ---
Occupation: Retired / Disability.
Living Situation: Lives with spouse. Split-level home (stairs are a barrier).
Habits: 
    - Smoking: Quit 10 years ago (20 pack-year history).
    - Alcohol: Occasional glass of wine.
    - Diet: Tries to follow low-sodium, but compliance varies.

--- FAMILY HISTORY ---
Mother: T2DM, HTN. Died age 78 (Stroke).
Father: COPD, CAD. Died age 72 (MI).
Siblings: Brother with similar autoimmune condition.

--- CURRENT MEDICATIONS (Reconciled) ---
1. {condition} Specific Agent: High dose / biologic.
2. Lisinopril 20mg Daily (HTN).
3. Atorvastatin 40mg Daily (Lipids).
4. Metformin 1000mg BID (Metabolic).
5. Omeprazole 20mg Daily (GERD).
6. Multivitamin Daily.
7. Vitamin D3 2000 IU Daily.

--- PLAN ---
1. Continue current biologic/therapy.
2. Monitor renal function every 3 months.
3. Referral to PT for mobility/strength.
4. Dietary consult for inflammation reduction.
    """
    # Repeat/Extend to make it "Long as possible" per user request
    # We will duplicate the history section to simulate "more notes"
    extended = base + "\n\n" + "-"*30 + "\n[ARCHIVED NOTES - 2015-2017]\n" + base.replace("2018", "2015").replace("2019", "2016").replace("2020", "2017")
    
    return extended

# --- Data Definitions (14 Complex Patients) ---

COMPLEX_PATIENTS = [
    # 1. Cardiology (John Doe)
    {
        "full_name": "John Doe",
        "dob": "1965-04-12",
        "notes": "Complex cardiology case. Post-MI with renal complications.",
        "lifestyle": {"diet": "High Sodium", "exercise": "Sedentary", "smoking": "Former"},
        "genetics": {"findings": [{"gene": "APOL1", "variant": "G1/G2", "impact": "High Risk CKD"}]},
        "documents": [
            {
                "filename": "Full_Medical_History.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("John Doe", "Coronary Artery Disease & CKD", "History of STEMI 2018, 3 stents placed. Recurrent angina."),
                "structured": {"diagnoses": [{"condition": "CAD"}, {"condition": "CKD Stage 3"}], "medications": [{"name": "Lisinopril", "dosage": "20mg"}, {"name": "Atorvastatin", "dosage": "40mg"}]}
            }
        ]
    },
    # 2. Rheumatology (Maria Rodriguez)
    {
        "full_name": "Maria Rodriguez",
        "dob": "1978-11-20",
        "notes": "Ankylosing Spondylitis. HLA-B27 Positive.",
        "lifestyle": {"diet": "Vegan", "exercise": "Yoga"},
        "genetics": {"findings": [{"gene": "HLA-B", "variant": "B27", "impact": "Positive"}]},
        "documents": [
             {
                "filename": "Rheum_History_Full.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Maria Rodriguez", "Ankylosing Spondylitis", "Failed NSAIDs. On Humira. Axial and peripheral involvement."),
                "structured": {"diagnoses": [{"condition": "Ankylosing Spondylitis"}], "medications": [{"name": "Humira"}]}
            }
        ]
    },
    # 3. Oncology (Robert Chen)
    {
        "full_name": "Robert Chen",
        "dob": "1955-08-30",
        "notes": "Non-Small Cell Lung Cancer (NSCLC). EGFR Exon 19 Deletion.",
        "lifestyle": {"smoking": "Current (1ppd)", "diet": "Low Appetite"},
        "genetics": {"findings": [{"gene": "EGFR", "variant": "Exon 19 Del", "impact": "Osimertinib Sensitive"}]},
        "documents": [
             {
                "filename": "Oncology_Records.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Robert Chen", "Metastatic NSCLC", "Brain mets treated with SRS. Lung primary stable on Tagrisso."),
                "structured": {"diagnoses": [{"condition": "NSCLC Stage IV"}], "medications": [{"name": "Osimertinib"}]}
            }
        ]
    },
    # 4. Neurology (Emily Blunt)
    {
        "full_name": "Emily Blunt",
        "dob": "1990-02-14",
        "notes": "Relapsing-Remitting Multiple Sclerosis (RRMS).",
        "lifestyle": {"exercise": "Active", "stress": "High"},
        "genetics": {},
        "documents": [
             {
                "filename": "Neurology_File.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Emily Blunt", "RRMS", "History of Optic Neuritis. JC Virus positive. On Tysabri -> Ocrevus switch."),
                "structured": {"diagnoses": [{"condition": "Multiple Sclerosis"}], "medications": [{"name": "Ocrelizumab"}]}
            }
        ]
    },
    # 5. Endocrinology (Michael Johnson)
    {
        "full_name": "Michael Johnson",
        "dob": "2005-06-22",
        "notes": "Type 1 Diabetes Mellitus. Insulin Pump user.",
        "lifestyle": {"diet": "Carb Counting", "exercise": "Varsity Soccer"},
        "genetics": {},
        "documents": [
             {
                "filename": "Endo_Summary.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Michael Johnson", "Type 1 Diabetes", "Dx age 8. DKA admission 2019. Pump user (T-Slim). Hypoglycemia unawareness."),
                "structured": {"diagnoses": [{"condition": "Type 1 Diabetes"}], "medications": [{"name": "Insulin Aspart"}]}
            }
        ]
    },
    # 6. Psychiatry (Sarah Connor)
    {
        "full_name": "Sarah Connor",
        "dob": "1982-05-18",
        "notes": "PTSD and Major Depressive Disorder.",
        "lifestyle": {"sleep": "Insomnia", "stress": "Severe"},
        "genetics": {"findings": [{"gene": "CYP2D6", "variant": "*4/*4", "impact": "Poor Metabolizer"}]},
        "documents": [
             {
                "filename": "Psych_History.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Sarah Connor", "Complex PTSD & TRD", "Multiple medication trials failed. CYP2D6 issues. Processing trauma therapy."),
                "structured": {"diagnoses": [{"condition": "PTSD"}, {"condition": "MDD"}], "medications": [{"name": "Sertraline"}]}
            }
        ]
    },
    # 7. Infectious Disease (David Kim)
    {
        "full_name": "David Kim",
        "dob": "1995-03-10",
        "notes": "HIV Controlled. Hepatitis B confection.",
        "lifestyle": {},
        "genetics": {},
        "documents": [
             {
                "filename": "ID_Records.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("David Kim", "HIV/Hep B Coinfection", "Undetectable viral load. Renal function monitoring due to Tenofovir."),
                "structured": {"diagnoses": [{"condition": "HIV"}, {"condition": "Hep B"}], "medications": [{"name": "Biktarvy"}]}
            }
        ]
    },
    # 8. Geriatrics (Linda Hamilton)
    {
        "full_name": "Linda Hamilton",
        "dob": "1940-09-26",
        "notes": "Early Alzheimer's. Recurrent Falls.",
        "lifestyle": {"mobility": "Walker use"},
        "genetics": {"findings": [{"gene": "APOE", "variant": "e4/e4", "impact": "High Risk Alzheimer's"}]},
        "documents": [
             {
                "filename": "Geriatric_Care_File.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Linda Hamilton", "Alzheimer's Dementia", "MMSE 22. Behavioral issues (sundowning). Caregiver burnout concern."),
                "structured": {"diagnoses": [{"condition": "Alzheimer's"}], "medications": [{"name": "Donepezil"}, {"name": "Memantine"}]}
            }
        ]
    },
    # 9. Trauma/Surgery (James Bond)
    {
        "full_name": "James Bond",
        "dob": "1975-04-13",
        "notes": "Post-Op Right Femur Fracture and Splenectomy.",
        "lifestyle": {"smoking": "Never", "alcohol": "Heavy"},
        "genetics": {},
        "documents": [
             {
                "filename": "Trauma_Surg_Summary.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("James Bond", "Polytrauma", "MVA 2024. Femur ORIF. Splenectomy. Post-op wound infection (resolved)."),
                "structured": {"diagnoses": [{"condition": "Femur Fracture"}, {"condition": "Asplenia"}], "medications": [{"name": "Oxycodone"}]}
            }
        ]
    },
    # 10. GI (Fiona Gallagher)
    {
        "full_name": "Fiona Gallagher",
        "dob": "1998-12-05",
        "notes": "Crohn's Disease. Fistulizing phenotype.",
        "lifestyle": {"diet": "Low Residue", "smoking": "Current"},
        "genetics": {"findings": [{"gene": "NOD2", "variant": "Mutated", "impact": "Crohn's Risk"}]},
        "documents": [
             {
                "filename": "GI_Records_Crohns.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Fiona Gallagher", "Fistulizing Crohn's", "Perianal disease. Multiple abscess drainages. Stelara failure. Remicade start."),
                "structured": {"diagnoses": [{"condition": "Crohn's Disease"}], "medications": [{"name": "Infliximab"}]}
            }
        ]
    },
    # 11. Pulmonology (Arthur Morgan - formerly Patient A)
    {
        "full_name": "Arthur Morgan",
        "dob": "1963-11-04",
        "notes": "Severe COPD. Chronic Hypoxic Respiratory Failure.",
        "lifestyle": {"smoking": "Current", "environmental": "Coal Dust"},
        "genetics": {"findings": [{"gene": "SERPINA1", "variant": "Normal", "impact": "No A1AT Deficiency"}]},
        "documents": [
             {
                "filename": "Pulm_Complex_History.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Arthur Morgan", "COPD GOLD 4", "Home O2 dependent (2L). Frequent exacerbations (3x/year). Cor Pulmonale."),
                "structured": {"diagnoses": [{"condition": "COPD"}, {"condition": "Cor Pulmonale"}], "medications": [{"name": "Trelegy"}, {"name": "Prednisone"}]}
            }
        ]
    },
    # 12. Hematology (Beatrice Kiddo - formerly Patient B)
    {
        "full_name": "Beatrice Kiddo",
        "dob": "1992-06-03",
        "notes": "Sickle Cell Disease (HbSS).",
        "lifestyle": {},
        "genetics": {"findings": [{"gene": "HBB", "variant": "HbS/HbS", "impact": "Sickle Cell"}]},
        "documents": [
             {
                "filename": "Heme_Sickle_Cell.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Beatrice Kiddo", "Sickle Cell Disease", "Acute Chest Syndrome hx. Avascular Necrosis of femoral head. Chronic pain."),
                "structured": {"diagnoses": [{"condition": "Sickle Cell Disease"}], "medications": [{"name": "Hydroxyurea"}, {"name": "Morphine"}]}
            }
        ]
    },
    # 13. Nephrology (Charles Xavier - formerly Patient C)
    {
        "full_name": "Charles Xavier",
        "dob": "1950-07-13",
        "notes": "ESRD on Hemodialysis.",
        "lifestyle": {"diet": "Renal"},
        "genetics": {},
        "documents": [
             {
                "filename": "Nephro_Dialysis_Rec.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Charles Xavier", "ESRD", "Dialysis MWF. AV Fistula L arm. Secondary Hyperparathyroidism uncontrolled."),
                "structured": {"diagnoses": [{"condition": "ESRD"}], "medications": [{"name": "Sevelamer"}, {"name": "Cinacalcet"}]}
            }
        ]
    },
    # 14. Dermatology (Diana Prince - formerly Patient D)
    {
        "full_name": "Diana Prince",
        "dob": "1985-03-22",
        "notes": "Severe Psoriasis + PsA.",
        "lifestyle": {"stress": "Moderate"},
        "genetics": {},
        "documents": [
             {
                "filename": "Derm_Complex.txt",
                "content_type": "text/plain",
                "content": generate_complex_history("Diana Prince", "Psoriatic Arthritis", "PASI > 15. Dactylitis. Methotrexate intolerance. Cosentyx effective."),
                "structured": {"diagnoses": [{"condition": "Psoriasis"}, {"condition": "Psoriatic Arthritis"}], "medications": [{"name": "Secukinumab"}]}
            }
        ]
    }
]

def export_to_files(base_dir: str):
    """Write the mock documents to the local filesystem in data/ folder."""
    data_path = Path(base_dir) / "data"
    data_path.mkdir(exist_ok=True)
    
    logger.info(f"Exporting files to {data_path}...")
    
    for p in COMPLEX_PATIENTS:
        p_name = p["full_name"].replace(" ", "_")
        p_dir = data_path / p_name
        p_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in p["documents"]:
            filename = doc["filename"]
            content = doc["content"].strip()
            # If we wanted to simulate CSV we could change extension, but content is text.
            # Using implicit writing.
            file_path = p_dir / filename
            with open(file_path, "w") as f:
                f.write(content)
                
    logger.info("Filesystem export complete.")

def clean_database(conn, tenant_id=None):
    logger.info("Cleaning existing sample data...")
    names = [p["full_name"] for p in COMPLEX_PATIENTS]
    if names:
        placeholders = ",".join(["%s"] * len(names))
        conn.execute(f"DELETE FROM patients WHERE full_name IN ({placeholders})", tuple(names))
        conn.commit()

def seed_data():
    settings = get_settings()
    logger.info("Starting data seed...")
    
    # 1. Export Files first
    export_to_files(str(REPO_ROOT))
    
    # 2. Seed DB
    with get_conn() as conn:
        clean_database(conn)
        
        # Check for tenant_id column
        try:
            check_col = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'patients' AND column_name = 'tenant_id'"
            ).fetchone()
            has_tenant_id = bool(check_col)
        except Exception:
            has_tenant_id = False
        
        default_tenant_id = None
        if has_tenant_id:
            logger.info("Detected tenant_id column. Using default system tenant.")
            default_tenant_id = "00000000-0000-0000-0000-000000000000"
            conn.execute(
                """
                INSERT INTO tenants (id, name, created_at)
                VALUES (%s, 'System Tenant', NOW())
                ON CONFLICT (id) DO NOTHING
                """,
                (default_tenant_id,)
            )
            conn.commit()

        for p_data in COMPLEX_PATIENTS:
            logger.info(f"Seeding patient: {p_data['full_name']}")
            
            # Create Patient
            # Construct insert dynamically or via if/else
            insert_cols = ["full_name", "dob", "notes", "lifestyle", "genetics"]
            insert_vals = [
                p_data["full_name"], 
                p_data["dob"], 
                p_data["notes"], 
                json.dumps(p_data.get("lifestyle", {})),
                json.dumps(p_data.get("genetics", {}))
            ]
            placeholders = ["%s", "%s", "%s", "%s", "%s"]
            
            if has_tenant_id:
                insert_cols.append("tenant_id")
                insert_vals.append(default_tenant_id)
                placeholders.append("%s")
                
            query = f"""
                INSERT INTO patients ({', '.join(insert_cols)})
                VALUES ({', '.join(placeholders)})
                RETURNING id
            """
            
            row = conn.execute(query, tuple(insert_vals)).fetchone()
            patient_id = str(row["id"])
            
            # Process Documents
            for doc_data in p_data["documents"]:
                filename = doc_data["filename"]
                raw_text = doc_data["content"].strip()
                content_type = doc_data["content_type"]
                structured_data = doc_data.get("structured", {})
                
                logger.info(f"  - Document: {filename}")
                
                # Mock storage upload
                storage_path = f"{patient_id}/{uuid4()}_{filename}"
                try:
                    upload_bytes(settings.storage_bucket, storage_path, raw_text.encode("utf-8"), content_type)
                except Exception as e:
                    logger.warning(f"Failed to upload to storage: {e}")

                # Insert Document
                doc_row = conn.execute(
                    """
                    INSERT INTO documents (patient_id, filename, content_type, storage_path)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (patient_id, filename, content_type, storage_path)
                ).fetchone()
                document_id = str(doc_row["id"])
                
                # Insert Extraction
                extraction_row = conn.execute(
                    """
                    INSERT INTO extractions (document_id, raw_text, structured)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (document_id, raw_text, json.dumps(structured_data))
                ).fetchone()
                extraction_id = extraction_row["id"]
                
                # Insert Structured Tables (Labs, Meds, Diagnoses)
                # Labs
                for lab in structured_data.get("labs", []):
                    # Check if table exists (lazy check)
                    try:
                        conn.execute(
                            """
                            INSERT INTO lab_results 
                            (patient_id, extraction_id, test_name, value, unit, flag, reference_range, test_date, panel)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                patient_id, extraction_id,
                                lab.get("test_name"), lab.get("value"), lab.get("unit"),
                                lab.get("flag"), lab.get("reference_range"), 
                                lab.get("date"), lab.get("panel")
                            )
                        )
                    except Exception: pass
                
                # Meds
                for med in structured_data.get("medications", []):
                    try:
                        conn.execute(
                            """
                            INSERT INTO medications
                            (patient_id, extraction_id, name, dosage, frequency, route, start_date, end_date, status)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                patient_id, extraction_id,
                                med.get("name"), med.get("dosage"), med.get("frequency"),
                                med.get("route"), med.get("start_date"), med.get("end_date"),
                                med.get("status", "active")
                            )
                        )
                    except Exception: pass

                # Diagnoses
                for dx in structured_data.get("diagnoses", []):
                    try:
                        conn.execute(
                            """
                            INSERT INTO diagnoses
                            (patient_id, extraction_id, condition, code, status, date_onset)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                patient_id, extraction_id,
                                dx.get("condition"), dx.get("code"), dx.get("status"),
                                dx.get("date_onset")
                            )
                        )
                    except Exception: pass
                
                # Embeddings
                try:
                    logger.info("    Generating embeddings...")
                    chunks = chunk_text(raw_text)
                    if chunks:
                        texts = [c["chunk_text"] for c in chunks]
                        vectors = []
                        try:
                            vectors = embed_texts(texts)
                        except Exception:
                            logger.warning("    Embedding API failed/not configured. Inserting zero-vectors.")
                            vectors = [[0.0] * 3072 for _ in texts]
                        
                        if vectors:
                            for chunk, vector in zip(chunks, vectors):
                                conn.execute(
                                    """
                                    INSERT INTO embeddings (document_id, extraction_id, chunk_index, chunk_start, chunk_end, chunk_text, embedding)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """,
                                    (
                                        document_id, extraction_id,
                                        chunk["chunk_index"], chunk["chunk_start"], chunk["chunk_end"],
                                        chunk["chunk_text"], vector
                                    )
                                )
                except Exception as e:
                    logger.error(f"    Error processing embeddings: {e}")
                    
            conn.commit()
    
    logger.info("Data seed complete.")

if __name__ == "__main__":
    seed_data()
