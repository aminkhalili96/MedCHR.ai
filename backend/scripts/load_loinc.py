"""
LOINC Database Loader Script

Loads LOINC codes and generates embeddings for vector-based auto-coding.
Run this script after downloading LoincTableCore.csv from loinc.org

Gap Reference: T01, T02
"""

import csv
import os
import sys
from typing import Generator

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Common LOINC codes for MVP (subset for quick setup without full LOINC download)
COMMON_LOINC_CODES = [
    # Basic Metabolic Panel
    {"code": "2951-2", "component": "Sodium", "long_common_name": "Sodium [Moles/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2823-3", "component": "Potassium", "long_common_name": "Potassium [Moles/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2075-0", "component": "Chloride", "long_common_name": "Chloride [Moles/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2028-9", "component": "Carbon dioxide", "long_common_name": "Carbon dioxide, total [Moles/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "3094-0", "component": "Urea nitrogen", "long_common_name": "Urea nitrogen [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2160-0", "component": "Creatinine", "long_common_name": "Creatinine [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2345-7", "component": "Glucose", "long_common_name": "Glucose [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "33914-3", "component": "eGFR", "long_common_name": "Glomerular filtration rate/1.73 sq M.predicted", "system": "Ser/Plas"},
    
    # Lipid Panel
    {"code": "2093-3", "component": "Cholesterol", "long_common_name": "Cholesterol [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2571-8", "component": "Triglycerides", "long_common_name": "Triglyceride [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2085-9", "component": "HDL", "long_common_name": "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2089-1", "component": "LDL", "long_common_name": "Cholesterol in LDL [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    
    # CBC
    {"code": "718-7", "component": "Hemoglobin", "long_common_name": "Hemoglobin [Mass/volume] in Blood", "system": "Bld"},
    {"code": "4544-3", "component": "Hematocrit", "long_common_name": "Hematocrit [Volume Fraction] of Blood", "system": "Bld"},
    {"code": "6690-2", "component": "WBC", "long_common_name": "Leukocytes [#/volume] in Blood", "system": "Bld"},
    {"code": "777-3", "component": "Platelets", "long_common_name": "Platelets [#/volume] in Blood", "system": "Bld"},
    {"code": "789-8", "component": "RBC", "long_common_name": "Erythrocytes [#/volume] in Blood", "system": "Bld"},
    
    # Liver Function
    {"code": "1742-6", "component": "ALT", "long_common_name": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "1920-8", "component": "AST", "long_common_name": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "6768-6", "component": "ALP", "long_common_name": "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "1975-2", "component": "Bilirubin", "long_common_name": "Bilirubin.total [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2885-2", "component": "Protein", "long_common_name": "Protein [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "1751-7", "component": "Albumin", "long_common_name": "Albumin [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    
    # Thyroid
    {"code": "3016-3", "component": "TSH", "long_common_name": "Thyrotropin [Units/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "3026-2", "component": "T4 Free", "long_common_name": "Thyroxine (T4) free [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "3053-6", "component": "T3 Free", "long_common_name": "Triiodothyronine (T3) Free [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    
    # Cardiac
    {"code": "10839-9", "component": "Troponin I", "long_common_name": "Troponin I.cardiac [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "42757-5", "component": "Troponin T", "long_common_name": "Troponin T.cardiac [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "30934-4", "component": "BNP", "long_common_name": "Natriuretic peptide B [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    
    # Coagulation
    {"code": "5902-2", "component": "PT", "long_common_name": "Prothrombin time (PT)", "system": "Bld"},
    {"code": "6301-6", "component": "INR", "long_common_name": "INR in Platelet poor plasma by Coagulation assay", "system": "PPP"},
    {"code": "3173-2", "component": "PTT", "long_common_name": "Activated partial thromboplastin time (aPTT) in Blood", "system": "Bld"},
    
    # Diabetes
    {"code": "4548-4", "component": "HbA1c", "long_common_name": "Hemoglobin A1c/Hemoglobin.total in Blood", "system": "Bld"},
    
    # Iron
    {"code": "2498-4", "component": "Iron", "long_common_name": "Iron [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2500-7", "component": "Ferritin", "long_common_name": "Ferritin [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2502-3", "component": "TIBC", "long_common_name": "Iron binding capacity [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    
    # Vitamins
    {"code": "2132-9", "component": "Vitamin B12", "long_common_name": "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2284-8", "component": "Folate", "long_common_name": "Folate [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "1989-3", "component": "Vitamin D", "long_common_name": "25-Hydroxyvitamin D3 [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    
    # Electrolytes
    {"code": "17861-6", "component": "Calcium", "long_common_name": "Calcium [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2601-3", "component": "Magnesium", "long_common_name": "Magnesium [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    {"code": "2777-1", "component": "Phosphorus", "long_common_name": "Phosphate [Mass/volume] in Serum or Plasma", "system": "Ser/Plas"},
    
    # Urinalysis
    {"code": "5811-5", "component": "Specific Gravity", "long_common_name": "Specific gravity of Urine by Test strip", "system": "Urine"},
    {"code": "5803-2", "component": "pH", "long_common_name": "pH of Urine by Test strip", "system": "Urine"},
    {"code": "5804-0", "component": "Protein Urine", "long_common_name": "Protein [Mass/volume] in Urine by Test strip", "system": "Urine"},
    {"code": "5792-7", "component": "Glucose Urine", "long_common_name": "Glucose [Mass/volume] in Urine by Test strip", "system": "Urine"},
]


def load_common_loinc(conn):
    """Load common LOINC codes into database."""
    print(f"Loading {len(COMMON_LOINC_CODES)} common LOINC codes...")
    
    for loinc in COMMON_LOINC_CODES:
        conn.execute("""
            INSERT INTO ref_loinc (code, component, long_common_name, system)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (code) DO UPDATE SET
                component = EXCLUDED.component,
                long_common_name = EXCLUDED.long_common_name,
                system = EXCLUDED.system
        """, (loinc["code"], loinc["component"], loinc["long_common_name"], loinc["system"]))
    
    conn.commit()
    print(f"Loaded {len(COMMON_LOINC_CODES)} LOINC codes")


def generate_loinc_embeddings(conn):
    """Generate embeddings for LOINC codes using OpenAI."""
    from app.embeddings import embed_texts
    from pgvector.psycopg import Vector
    
    # Get codes without embeddings
    rows = conn.execute("""
        SELECT l.code, l.long_common_name, l.component
        FROM ref_loinc l
        LEFT JOIN ref_loinc_embeddings le ON le.code = l.code
        WHERE le.code IS NULL
    """).fetchall()
    
    if not rows:
        print("All LOINC codes already have embeddings")
        return
    
    print(f"Generating embeddings for {len(rows)} LOINC codes...")
    
    # Batch processing
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        texts = [f"{r['component']}: {r['long_common_name']}" for r in batch]
        embeddings = embed_texts(texts)
        
        for row, embedding in zip(batch, embeddings):
            conn.execute("""
                INSERT INTO ref_loinc_embeddings (code, embedding)
                VALUES (%s, %s)
                ON CONFLICT (code) DO UPDATE SET embedding = EXCLUDED.embedding
            """, (row["code"], Vector(embedding)))
        
        conn.commit()
        print(f"  Processed {min(i+batch_size, len(rows))}/{len(rows)}")
    
    print("Embedding generation complete")


def load_from_csv(csv_path: str, conn):
    """Load LOINC codes from official LoincTableCore.csv file."""
    if not os.path.exists(csv_path):
        print(f"LOINC CSV not found: {csv_path}")
        print("Download from https://loinc.org/downloads/")
        return
    
    print(f"Loading LOINC from {csv_path}...")
    count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conn.execute("""
                INSERT INTO ref_loinc (code, component, long_common_name, system)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO NOTHING
            """, (
                row.get("LOINC_NUM"),
                row.get("COMPONENT"),
                row.get("LONG_COMMON_NAME"),
                row.get("SYSTEM")
            ))
            count += 1
            if count % 1000 == 0:
                conn.commit()
                print(f"  Loaded {count} codes...")
    
    conn.commit()
    print(f"Loaded {count} LOINC codes from CSV")


def main():
    """Main entry point."""
    import argparse
    from app.db import get_conn
    
    parser = argparse.ArgumentParser(description="Load LOINC codes into database")
    parser.add_argument("--csv", help="Path to LoincTableCore.csv")
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings")
    parser.add_argument("--common-only", action="store_true", help="Load only common codes")
    args = parser.parse_args()
    
    with get_conn() as conn:
        if args.csv:
            load_from_csv(args.csv, conn)
        elif args.common_only:
            load_common_loinc(conn)
        else:
            # Default: load common codes
            load_common_loinc(conn)
        
        if args.embeddings:
            generate_loinc_embeddings(conn)
    
    print("Done!")


if __name__ == "__main__":
    main()
