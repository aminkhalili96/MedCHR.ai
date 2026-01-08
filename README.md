# MedCHR MVP

Clinician-in-the-loop pipeline for turning raw patient records into editable Client Health Reports (CHRs).

## Quick Start
Run all commands from the project root (one level up from this file).

1) Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2) Copy env template:

```bash
cp .env.example .env
```

Fill in:
- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `APP_SECRET_KEY`
- `APP_USERNAME`
- `APP_PASSWORD`

3) Create Supabase Storage bucket:
- **Storage → New bucket →** `medchr-uploads` (private)

4) Enable pgvector in Supabase:
- **Database → Extensions →** `vector`

5) Apply DB schema:

```bash
python -m backend.scripts.init_db
```

6) Run API:

```bash
uvicorn app.main:app --app-dir backend --reload
```

7) One-command dev run (applies schema + starts API):

```bash
./run_dev.sh
```

8) Open UI:
- Visit `http://127.0.0.1:8000/ui`
- Login with `APP_USERNAME` / `APP_PASSWORD`
- Embeddings debug: `http://127.0.0.1:8000/ui/embeddings`
- RAG viewer: open a patient and click “View RAG Top-K Chunks”

## Mock Data Import (Bulk)
To load the synthetic dataset into Postgres and Supabase Storage:

```bash
python -m backend.scripts.import_mock_data
```

Optional flags:
- `--patient patient_a` (repeat for multiple)
- `--skip-embed` (skip embeddings)
- `--skip-draft` (skip CHR draft)

## Key Endpoints
- `POST /patients` — create patient
- `POST /patients/{patient_id}/documents` — upload document
- `POST /documents/{document_id}/extract` — extract raw text + structured data
- `POST /documents/{document_id}/embed` — store embeddings in pgvector
- `POST /chr/draft` — generate CHR draft

## OCR Notes
This MVP uses Tesseract. Install it (macOS):

```bash
brew install tesseract
```

## Project Docs
- `PROJECT_DOC.md` — scope, plan, interview prep
- `WORKFLOW.drawio` — workflow diagram source of truth
