# Implementation Plan: AI Enhancements (Phase 4)

## Objective
Maximize clinical accuracy and minimize hallucinations through advanced AI orchestration.

## 1. Advanced RAG & Grounding
**Gap**: Basic RAG risks missing context.

*   **Hybrid Search**: Combine dense retrieval (embedding) with sparse retrieval (BM25/Keyword) to catch specific acronyms or dosages that embeddings might miss.
*   **Parent-Child Indexing**: Store small chunks (for search matching) but retrieve larger parent chunks (for LLM context) to ensure full sentences/paragraphs are provided.

## 2. "Auditor" Agent (Self-Correction)
**Gap**: No verification of LLM output.

*   **Workflow**:
    1.  Generator LLM drafts CHR section.
    2.  Auditor LLM receives: {Source Chunks} + {Draft}.
    3.  Prompt: "Verify every claim in the Draft against Source. Return 'Verified' or 'Hallucination' with correction."
    4.  Logic: If discrepancies found, regenerate.

## 3. Confidence & Uncertainty Quantification
**Gap**: Clinician doesn't know what the AI is unsure about.

*   **Implementation**:
    *   Ask LLM to output `<confidence_score>` for extracted values.
    *   **Logprobs**: If using OpenAI, check logprobs for token probability.
    *   **UI**: Highlight low-confidence sections in Red/Yellow for mandatory human review.

## 4. Layout-Aware Parsing (Vision)
**Gap**: Tables often break in text extraction.

*   **Pipeline**:
    1.  Convert PDF page to Image.
    2.  Use Vision Model (GPT-4o / LLaVA) or Document Intelligence (Azure/AWS) to identify layout regions (Header, Table, Footer).
    3.  Extract Table data *as a grid* specifically, preserving row/column relationships.

## Roadmap Tasks
- [ ] Backend: Refactor `rag.py` to support Hybrid Search (requires pgvector+pg_search or external vector DB like Qdrant/Pinecone, but pgvector hybrid is possible with custom SQL).
- [ ] Backend: Implement `Auditor` class in `chr.py`.
- [ ] Backend: Integrate Vision API for PDF-to-Image-to-Text flow in `ocr.py`.
- [ ] Frontend: Add UI support for "Confidence Highlights".
