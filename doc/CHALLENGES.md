# MedCHR Project: Challenges & Solutions (Interview Prep)

> **Context**: Building a medical record summarization tool is harder than standard text summarization because accuracy is life-critical and data is messy. Here are the top challenges you will face and how to solve them.

---

## 1. The "Messy Data" Challenge (OCR & Parsing)
**The Problem**: Medical records are rarely clean JSON. They are often:
- Scanned PDFs (images of text).
- Faxed copies (low resolution, noise, tilted).
- Handwritten clinical notes.
- Multi-column layouts (tables that break across pages).

**The Solution**:
- **Hybrid OCR Pipeline**: Use **Tesseract** for clean typed text (fast/cheap) but fallback to **Vision LLMs (GPT-4o Vision)** for complex layouts or handwriting.
- **Layout-Aware Parsing**: Don't just extract text blobs. Use tools like `LayoutParser` or `Azure Document Intelligence` to respect table structures.
- **Confidence Scoring**: Flag low-confidence OCR regions for human review. If the OCR isn't sure, don't guess—highlight it for the doctor.

**Interview Talking Point**: *"I realized standard OCR libraries fail on complex lab tables. I solved this by treating the page as an image first to detect layout regions, then applying targeted OCR to each section."*

---

## 2. The "Hallucination" Challenge (Safety)
**The Problem**: Generative AI (LLMs) loves to hallucinate. In healthcare, hallucinating a diagnosis or a medication dosage is dangerous.
- *Example*: The LLM reads "No diabetes" but summarizes it as "Diabetes".

**The Solution**:
- **Strict Grounding (RAG)**: The LLM must *never* generate facts from its training data. It must only extract facts present in the provided text partitions.
- **Citation Requirement**: Force the model to output a `source_id` or `line_number` for every claim. If it can't cite a source, it can't include the fact.
- **Self-Verification Step**: Implement a second LLM pass that acts as a "Auditor". It reads the generated summary and the source text and asks: "Is this claim supported by the source?"

**Interview Talking Point**: *"To prevent hallucinations, I implemented a strict citation mechanism. Every sentence in the generated report must link back to a specific highlight in the original PDF, allowing the clinician to verify it instantly."*

---

## 3. The "normalization" Challenge (Interoperability)
**The Problem**: Different labs use different names and units.
- Lab A: `Glucose, Fasting` (mg/dL)
- Lab B: `Fasting Blood Sugar` (mmol/L)
- Lab C: `Glu`
A simple string match fails. You need to map these to a single standard.

**The Solution**:
- **Standard Ontologies**: Map extracted entities to standard codes like **LOINC** (for labs) or **RxNorm** (for meds).
- **Unit Conversion Layer**: A robust logic layer that detects units and normalizes them (e.g., always store glucose as mg/dL in the DB).
- **Semantic Vector Search (pgvector)**: Instead of brittle string matching, use embeddings. "Glucose" and "Fasting Sugar" live close together in vector space. I use `pgvector` to find the nearest LOINC concept significantly more accurately than fuzzy text matching.

**Interview Talking Point**: *"Data heterogeneity is huge. I replaced standard regex matching with valid semantic search using pgvector. This allows the system to understand that 'Gluc' and 'Fasting Blood Sugar' refer to the same clinical concept without maintaining thousands of manual regex rules."*

---

## 4. The "Data Privacy" Challenge (HIPAA/PDPA)
**The Problem**: You cannot just send patient names and identifiable data to public APIs (like OpenAI) without a BAA (Business Associate Agreement) or privacy guarantees.

**The Solution**:
- **PII Redaction**: Before sending text to an LLM, run a local NER (Named Entity Recognition) model (like HuggingFace `presidio`) to strip names, dates, and IDs. Replace them with `<PATIENT_1>`.
- **Zero-Retention APIs**: Use "Enterprise" endpoints where the provider guarantees zero data retention for training.
- **Local LLMs (Future)**: Run sensitive extraction tasks on local models (Llama 3, Mistral) within your own VPC.

**Interview Talking Point**: *"Privacy is non-negotiable. I designed the architecture to redact PII locally before any cloud processing, and we only use stateless model endpoints that do not train on our data."*

---

## 5. The "Workflow" Challenge (Adoption)
**The Problem**: Doctors are busy. If your tool adds clicks, they won't use it. "Human-in-the-loop" sounds good but can be tedious if the UX is bad.

**The Solution**:
- **"Accept/Reject" UX**: Don't make them write. Make them review. Present the draft with "Accept" buttons.
- **Visual Evidence**: When they click a generated sentence, virtually scroll the PDF to the exact location where that data came from.
- **Progressive Trust**: Start by automating the boring stuff (demographics, medication lists) to build trust before tackling complex diagnostic summaries.

**Interview Talking Point**: *"The goal isn't to replace the doctor, but to remove the clerical burden. I focused on a 'Review-First' UX where the AI does the heavy lifting of drafting, and the doctor just acts as the editor and signatory."*
