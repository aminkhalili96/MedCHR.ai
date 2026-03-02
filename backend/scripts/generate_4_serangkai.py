import sys
import os
import random
import logging
import uuid
import json
from datetime import date, datetime, timedelta
from pathlib import Path

# Setup path to import backend modules
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.db import get_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vital types and typical ranges for different conditions
VITAL_TEMPLATES = {
    "Hypertension": {
        "Blood Pressure": {"sys_min": 130, "sys_max": 160, "dia_min": 85, "dia_max": 100},
    },
    "Type 2 Diabetes Mellitus": {
        "BMI": {"min": 25.0, "max": 35.0},
        "Weight": {"min": 75.0, "max": 100.0}
    },
    "Obesity": {
        "BMI": {"min": 30.0, "max": 45.0},
        "Weight": {"min": 90.0, "max": 140.0}
    },
    "COPD": {
        "O2 Saturation": {"min": 88, "max": 95}
    },
    "Heart Failure": {
        "Weight": {"min": 70.0, "max": 95.0} # Fluctuates with fluid retention
    },
    "Default": {
        "Blood Pressure": {"sys_min": 110, "sys_max": 125, "dia_min": 70, "dia_max": 80},
        "Heart Rate": {"min": 60, "max": 90},
        "BMI": {"min": 20.0, "max": 25.0},
        "Weight": {"min": 60.0, "max": 80.0},
        "Temperature": {"min": 36.5, "max": 37.2},
        "O2 Saturation": {"min": 96, "max": 100}
    }
}

LAB_TEMPLATES = {
    "Hypertension": [
        {"name": "Sodium", "unit": "mmol/L", "min": 136, "max": 145},
        {"name": "Potassium", "unit": "mmol/L", "min": 3.5, "max": 5.1},
        {"name": "Creatinine", "unit": "mg/dL", "min": 0.8, "max": 1.4}
    ],
    "Hyperlipidemia": [
        {"name": "Total Cholesterol", "unit": "mg/dL", "min": 200, "max": 280, "flag": "High"},
        {"name": "LDL", "unit": "mg/dL", "min": 130, "max": 190, "flag": "High"},
        {"name": "HDL", "unit": "mg/dL", "min": 35, "max": 45, "flag": "Low"},
        {"name": "Triglycerides", "unit": "mg/dL", "min": 150, "max": 300, "flag": "High"}
    ],
    "Type 2 Diabetes Mellitus": [
        {"name": "HbA1c", "unit": "%", "min": 6.5, "max": 9.5, "flag": "High"},
        {"name": "Fasting Glucose", "unit": "mg/dL", "min": 126, "max": 250, "flag": "High"}
    ],
    "CKD Stage 4": [
        {"name": "eGFR", "unit": "mL/min", "min": 15, "max": 29, "flag": "Low"},
        {"name": "Creatinine", "unit": "mg/dL", "min": 2.5, "max": 4.5, "flag": "High"},
        {"name": "BUN", "unit": "mg/dL", "min": 40, "max": 80, "flag": "High"}
    ],
    "Rheumatoid Arthritis": [
        {"name": "CRP", "unit": "mg/L", "min": 10, "max": 40, "flag": "High"},
        {"name": "ESR", "unit": "mm/hr", "min": 30, "max": 80, "flag": "High"}
    ],
    "Anemia of Chronic Disease": [
        {"name": "Hemoglobin", "unit": "g/dL", "min": 8.0, "max": 11.0, "flag": "Low"},
        {"name": "Hematocrit", "unit": "%", "min": 25, "max": 33, "flag": "Low"},
        {"name": "Iron", "unit": "mcg/dL", "min": 40, "max": 60, "flag": "Low"}
    ],
    "Default": [
        {"name": "WBC", "unit": "K/uL", "min": 4.5, "max": 11.0},
        {"name": "Hemoglobin", "unit": "g/dL", "min": 12.0, "max": 16.0},
        {"name": "Platelets", "unit": "K/uL", "min": 150, "max": 450}
    ]
}

FOUR_SERANGKAI = ["Hypertension", "Hyperlipidemia", "Type 2 Diabetes Mellitus", "Obesity"]

def generate_historical_reads(conn, patient_id: uuid.UUID, conditions: set, count: int = 10, days_back: int = 1095):
    """Generates a history of lab results and vitals for a patient."""
    today = datetime.now()
    
    # Determine base profiles
    v_profiles = []
    l_profiles = []
    
    has_specifics = False
    for condition in conditions:
        if condition in VITAL_TEMPLATES:
            v_profiles.append(VITAL_TEMPLATES[condition])
            has_specifics = True
        if condition in LAB_TEMPLATES:
            l_profiles.append(LAB_TEMPLATES[condition])
            has_specifics = True
            
    if not has_specifics:
        v_profiles.append(VITAL_TEMPLATES["Default"])
        l_profiles.append(LAB_TEMPLATES["Default"])
        
    for _ in range(count):
        # Pick a random date in the past
        record_date = today - timedelta(days=random.randint(1, days_back))
        
        # --- Generate Vitals ---
        vitals_to_add = {}
        for vp in v_profiles:
            for vt_name, vt_range in vp.items():
                vitals_to_add[vt_name] = vt_range  # Last one wins for overlapping
                
        # Fill in defaults for missing basic vitals
        for vt_name, vt_range in VITAL_TEMPLATES["Default"].items():
            if vt_name not in vitals_to_add:
                vitals_to_add[vt_name] = vt_range
                
        for v_name, v_range in vitals_to_add.items():
            if v_name == "Blood Pressure":
                sys = random.randint(v_range["sys_min"], v_range["sys_max"])
                dia = random.randint(v_range["dia_min"], v_range["dia_max"])
                conn.execute(
                    """
                    INSERT INTO vitals (patient_id, type, value_1, value_2, unit, recorded_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (patient_id, v_name, sys, dia, "mmHg", record_date)
                )
            else:
                val = round(random.uniform(v_range["min"], v_range["max"]), 1)
                unit = "kg/m2" if v_name == "BMI" else "kg" if v_name == "Weight" else "%" if v_name == "O2 Saturation" else "bpm" if v_name == "Heart Rate" else "C"
                conn.execute(
                    """
                    INSERT INTO vitals (patient_id, type, value_1, unit, recorded_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (patient_id, v_name, val, unit, record_date)
                )
                
        # --- Generate Labs ---
        labs_to_add = []
        for lp in l_profiles:
            labs_to_add.extend(lp)
            
        # Add basic CBC if not present
        if not any(l["name"] == "WBC" for l in labs_to_add):
            labs_to_add.extend(LAB_TEMPLATES["Default"])
            
        for lab in labs_to_add:
            val = round(random.uniform(lab["min"], lab["max"]), 2)
            flag = lab.get("flag", "Normal")
            # Sometimes normalise the flag if it's borderline
            if flag != "Normal" and random.random() < 0.2:
                flag = "Normal" # Sometimes they are controlled!
                
            conn.execute(
                """
                INSERT INTO lab_results (patient_id, test_name, value, unit, flag, reference_range, test_date, panel)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (patient_id, lab["name"], str(val), lab["unit"], flag, f'{lab["min"]}-{lab["max"]}', record_date, "Generated History")
            )
            
def run_simulation():
    logger.info("Starting historical data simulation for all patients...")
    with get_conn() as conn:
        # Get all patients
        patients = conn.execute("SELECT id FROM patients").fetchall()
        logger.info(f"Adding historical data for {len(patients)} patients.")

        for pt in patients:
            pid = pt['id'] if hasattr(pt, 'keys') else pt[0]
            
            # 1. Fetch current diagnoses
            existing_dx_records = conn.execute(
                "SELECT condition FROM diagnoses WHERE patient_id = %s", (pid,)
            ).fetchall()
            
            # Use appropriate logic to extract string conditions from the Row proxy
            existing_dx_list = []
            if existing_dx_records:
                for r in existing_dx_records:
                    if hasattr(r, 'keys'):
                       val = getattr(r, 'condition', r['condition'])
                       existing_dx_list.append(val)
                    else:
                       existing_dx_list.append(r[0])
            
            existing_dx = set(existing_dx_list)

            
            # 2. Add ALL 4 serangkai to ALL patients for this simulation
            conditions_to_ensure = set(existing_dx)
            for d in FOUR_SERANGKAI:
                if d not in existing_dx:
                    conn.execute(
                        """
                        INSERT INTO diagnoses (patient_id, condition, status, date_onset)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (pid, d, "Active", datetime.now() - timedelta(days=random.randint(365, 1800)))
                    )
                    conditions_to_ensure.add(d)
                    
            # 3. Generate History (5-15 encounters over the last 3 years)
            num_encounters = random.randint(5, 15)
            # Call generation script: it will generate info for BOTH original conditions and 4-serangkai
            generate_historical_reads(conn, pid, conditions_to_ensure, count=num_encounters, days_back=1095)
            
        conn.commit()
    logger.info("Historical simulation complete.")

if __name__ == "__main__":
    run_simulation()
