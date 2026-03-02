import sys
import os
import random
import logging
import uuid
import json
from datetime import date, datetime
from pathlib import Path

# Setup path to import backend modules
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.db import get_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MOCK_DOCS = [
    {
        "filename": "medchr_diabetes_labs.png",
        "doc_type": "Laboratory Report",
        "description": "Patient Lipid Panel and HbA1c Lab Report"
    },
    {
        "filename": "medchr_bp_reading.png",
        "doc_type": "Home Monitoring",
        "description": "Photo of Home Blood Pressure reading"
    },
     {
        "filename": "medchr_glucose_log.png",
        "doc_type": "Home Monitoring",
        "description": "Handwritten daily blood glucose log"
    },
     {
        "filename": "medchr_obesity_chart.png",
        "doc_type": "Clinical Note",
        "description": "EHR snapshot showing weight trend"
    }
]

def run_linking():
    logger.info("Starting to link mock documents to patients...")
    with get_conn() as conn:
        # Get all patients who have all 4 serangkai to ensure we link to the right ones
        # For simplicity, we just grab a random subset of 10 patients
        patients_rows = conn.execute("SELECT id FROM patients LIMIT 10").fetchall()
        
        for pt_row in patients_rows:
            pid = pt_row['id'] if hasattr(pt_row, 'keys') else pt_row[0]
            
            # Link 1-2 random mock documents
            docs_to_link = random.sample(MOCK_DOCS, k=random.randint(1, 2))
            
            for doc in docs_to_link:
                # In real scenario, this would be an S3/Supabase Storage path
                # For demo, we just point to the local mocked path
                s_path = f"mock_documents/{doc['filename']}"
                
                logger.info(f"Linking {doc['filename']} to patient {pid}")
                conn.execute(
                    """
                    INSERT INTO documents (patient_id, filename, content_type, storage_path, document_type)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (pid, doc["description"], "image/png", s_path, doc["doc_type"])
                )
                
        conn.commit()
    logger.info("Successfully linked mock documents.")

if __name__ == "__main__":
    run_linking()
