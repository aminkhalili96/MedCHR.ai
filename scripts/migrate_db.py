import sys
from pathlib import Path

# Add backend to path to import app modules
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.append(str(backend_path))

from app.db import get_conn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    schema_path = backend_path / "sql" / "schema.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found at {schema_path}")
        return

    sql = schema_path.read_text()
    
    logger.info("Applying schema changes...")
    try:
        with get_conn() as conn:
            conn.execute(sql)
            conn.commit()
        logger.info("Schema applied successfully.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        # If it failed, it might be partial execution or syntax error. 
        # Since we use IF NOT EXISTS, it should be idempotent mostly.
        # But adding columns to existing tables isn't handled by CREATE TABLE IF NOT EXISTS.
        # We might need explicit ALTER statements if tables exist.
        # For this prototype loop, we'll try to execute the raw SQL. 
        # If tables exist, CREATE TABLE IF NOT EXISTS does nothing.
        # We need to manually add columns if they are missing.
        manual_migration()

def manual_migration():
    logger.info("Attempting manual column additions...")
    with get_conn() as conn:
        # 1. Ensure tenants table exists (legacy DB might not have it)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              name TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        conn.commit()

        # Check/Add tenant_id to patients
        try:
            conn.execute("ALTER TABLE patients ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE")
            conn.commit()
            logger.info("Added tenant_id to patients")
        except Exception as e:
            logger.warning(f"Could not add tenant_id to patients: {e}")
            conn.rollback()

        # Check/Add users table
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                  email TEXT NOT NULL UNIQUE,
                  password_hash TEXT NOT NULL,
                  role TEXT NOT NULL DEFAULT 'clinician',
                  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not create users table: {e}")
            conn.rollback()

        # Check/Add tenant_id to audit_logs
        try:
            conn.execute("ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE")
            conn.commit()
            logger.info("Added tenant_id to audit_logs")
        except Exception as e:
            logger.warning(f"Could not add tenant_id to audit_logs: {e}")
            conn.rollback()

        # Create new clinical tables if they failed in main schema run
        for table_sql in [
            """
            CREATE TABLE IF NOT EXISTS medications (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
              extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
              name TEXT NOT NULL,
              dosage TEXT,
              frequency TEXT,
              route TEXT,
              start_date DATE,
              end_date DATE,
              status TEXT DEFAULT 'active',
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS lab_results (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
              extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
              test_name TEXT NOT NULL,
              value TEXT NOT NULL,
              unit TEXT,
              flag TEXT,
              reference_range TEXT,
              test_date DATE,
              panel TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS diagnoses (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
              extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
              condition TEXT NOT NULL,
              code TEXT,
              status TEXT,
              date_onset DATE,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        ]:
            try:
                conn.execute(table_sql)
                conn.commit()
            except Exception as e:
                logger.warning(f"Could not create clinical table: {e}")
                conn.rollback()
            
        # Create default tenant and user if none exist
        ensure_default_data(conn)

def ensure_default_data(conn):
    # Check if any tenant
    row = conn.execute("SELECT count(*) as cnt FROM tenants").fetchone()
    if row and row['cnt'] == 0:
        logger.info("Seeding default tenant and admin...")
        # Create default tenant
        t_row = conn.execute("INSERT INTO tenants (name) VALUES ('Demo Clinic') RETURNING id").fetchone()
        t_id = t_row['id']
        
        # Create default admin
        from app.auth import get_password_hash
        pw_hash = get_password_hash("admin") # Default password
        conn.execute(
            "INSERT INTO users (email, password_hash, role, tenant_id) VALUES (%s, %s, %s, %s)",
            ("admin@medchr.ai", pw_hash, "admin", t_id)
        )
        # Update existing data to own by this tenant
        conn.execute("UPDATE patients SET tenant_id = %s WHERE tenant_id IS NULL", (t_id,))
        # Audit logs might default to NULL, which is fine
        conn.commit()
        logger.info("Seeded default tenant (Demo Clinic) and user (admin@medchr.ai / admin)")

if __name__ == "__main__":
    run_migration()
