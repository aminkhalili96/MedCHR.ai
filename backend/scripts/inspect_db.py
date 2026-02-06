
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.db import get_conn
from backend.app.config import get_settings

def inspect():
    with get_conn() as conn:
        print("Inspecting 'tenants' table:")
        rows = conn.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'tenants'
            """
        ).fetchall()
        for r in rows:
            print(r)

if __name__ == "__main__":
    inspect()
