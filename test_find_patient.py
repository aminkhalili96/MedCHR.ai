import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.db import get_conn

def run():
    with get_conn() as conn:
        row = conn.execute("""
            SELECT patient_id FROM extractions LIMIT 1
        """).fetchone()
        if row:
            print("Patient with extraction:", row["patient_id"])
        else:
            print("No extractions found in the DB!")

if __name__ == "__main__":
    run()
