import psycopg
from app.config import get_settings

settings = get_settings()

with psycopg.connect(settings.database_url) as conn:
    with conn.cursor() as cur:
        print("Dropping embeddings table to rebuild dimensions")
        cur.execute("DROP TABLE IF EXISTS embeddings CASCADE;")
        
        cur.execute("""
        CREATE TABLE embeddings (
            id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
            tenant_id uuid REFERENCES tenants(id) ON DELETE CASCADE,
            document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
            extraction_id uuid REFERENCES extractions(id) ON DELETE CASCADE,
            chunk_index smallint NOT NULL,
            chunk_start integer NOT NULL,
            chunk_end integer NOT NULL,
            chunk_text text NOT NULL,
            embedding vector(1024) NOT NULL,
            created_at timestamptz DEFAULT CURRENT_TIMESTAMP,
            created_by varchar(255) DEFAULT 'system'
        );
        """)
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON embeddings(document_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_fts ON embeddings USING gin(to_tsvector('english', chunk_text));")
        
        print("Done fixing dimensions to 1024")
    conn.commit()
