import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.config import get_settings
from backend.app.main import _draft_chr
from backend.app.db import set_tenant_context, set_actor_context, get_conn

def run():
    patient_id = "3f01b711-18ba-4d09-a27d-12feb962c9df"
    
    settings = get_settings()
    with get_conn() as conn:
        row = conn.execute("SELECT tenant_id, id FROM users LIMIT 1").fetchone()
        tenant_id = str(row["tenant_id"])
        actor_id = str(row["id"])
    
    set_tenant_context(tenant_id)
    set_actor_context(actor_id)
    
    try:
        draft = _draft_chr(patient_id, notes="Test notes", actor="admin", tenant_id=tenant_id)
        print("Success!")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    run()
