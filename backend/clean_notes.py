import os
import psycopg
from dotenv import load_dotenv

# Load from .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

def clean_notes():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("Error: DATABASE_URL not found in environment or .env file.")
        return
        
    print("Connecting to DB to clean patient notes...")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, notes FROM patients WHERE notes LIKE 'Generated complex patient with:%'")
            rows = cur.fetchall()
            print(f"Found {len(rows)} patients with placeholder notes.")
            
            updated = 0
            for r in rows:
                new_note = r[1].replace("Generated complex patient with:", "Presenting Case:").strip()
                cur.execute("UPDATE patients SET notes = %s WHERE id = %s", (new_note, r[0]))
                updated += 1
                
            conn.commit()
            print(f"Successfully cleaned {updated} patient notes. They now look clinical.")

if __name__ == "__main__":
    clean_notes()
