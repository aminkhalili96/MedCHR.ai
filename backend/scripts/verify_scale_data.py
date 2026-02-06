#!/usr/bin/env python3
"""
verify_scale_data.py - Data Verification Script for MedCHR.ai

This script verifies that synthetic patient data has been successfully
generated and inserted into the database.

Usage:
    cd MedCHR.ai
    python -m backend.scripts.verify_scale_data

Output:
    Prints counts for patients, documents, labs, medications, diagnoses, and embeddings.
"""

import sys
from pathlib import Path
import logging

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Backend imports
from backend.app.db import get_conn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_data():
    with get_conn() as conn:
        logger.info("--- Data Verification ---")
        
        # Check Patients
        p_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()['count']
        logger.info(f"Total Patients: {p_count}")
        
        # Check Documents
        d_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()['count']
        logger.info(f"Total Documents: {d_count}")
        
        # Check Labs
        l_count = conn.execute("SELECT COUNT(*) FROM lab_results").fetchone()['count']
        logger.info(f"Total Lab Results: {l_count}")
        
        # Check Meds
        m_count = conn.execute("SELECT COUNT(*) FROM medications").fetchone()['count']
        logger.info(f"Total Medications: {m_count}")
        
        # Check Diagnoses
        dx_count = conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()['count']
        logger.info(f"Total Diagnoses: {dx_count}")
        
        # Check Embeddings
        e_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()['count']
        logger.info(f"Total Embeddings: {e_count}")

        # Basic Sanity Check
        if p_count >= 100:
            logger.info("SUCCESS: At least 100 patients found.")
        else:
            logger.error("FAILURE: Fewer than 100 patients found.")

        if e_count > 0:
            logger.info("SUCCESS: Embeddings generated.")
        else:
            logger.warning("WARNING: No embeddings found (Did API fail silently or disabled?)")

        # Check for Sample Integrity
        sample_pat = conn.execute("SELECT full_name, lifestyle FROM patients ORDER BY created_at DESC LIMIT 1").fetchone()
        logger.info(f"Sample Patient: {sample_pat['full_name']}")
        logger.info(f"Sample Lifestyle: {sample_pat['lifestyle']}")

if __name__ == "__main__":
    verify_data()
