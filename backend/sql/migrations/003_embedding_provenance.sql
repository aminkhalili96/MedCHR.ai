ALTER TABLE embeddings
  ADD COLUMN IF NOT EXISTS extraction_id UUID REFERENCES extractions(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS chunk_index INTEGER,
  ADD COLUMN IF NOT EXISTS chunk_start INTEGER,
  ADD COLUMN IF NOT EXISTS chunk_end INTEGER;

CREATE INDEX IF NOT EXISTS idx_embeddings_document_chunk ON embeddings(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_embeddings_extraction_id ON embeddings(extraction_id);
