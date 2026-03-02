# Mock Patients Data Plan (Demo-Rich, Synthetic)

Purpose: Create a synthetic, demo-rich dataset that exercises OCR, table parsing, normalization, embeddings/pgvector, RAG retrieval, CHR drafting, and audit logging.

All data is fully synthetic. Values are invented and not tied to any real person.
Branding uses a fictional hospital name and a simple logo graphic.
Reports are intentionally longer and more complex to simulate hospital-grade lab outputs.

## Dataset Overview
- 4 patients, 3–4 documents each (mixed PDF + scanned images).
- Mix of table-heavy labs, narrative notes, imaging reports, medication lists, and a genetics summary.
- At least 2 scanned images to trigger OCR.

## Patient A — Metabolic / Cardio Risk
Goal: Labs normalization, flags, medications, narrative context.

Documents:
1) **A1_Labs_Metabolic.pdf** (PDF, table-heavy)
   - Panels: Lipid panel, HbA1c, CMP, CBC
   - Fields: value, unit, reference range, abnormal flag
   - Tests: HDL, LDL, Total Cholesterol, Triglycerides, Glucose, HbA1c, AST, ALT, Creatinine
2) **A2_Progress_Note.txt** (text)
   - Symptoms: fatigue, weight gain
   - Lifestyle: sedentary, high sugar intake
   - Family history: CAD, T2D
3) **A3_Meds_List.pdf** (PDF or text)
   - Meds: metformin, atorvastatin
4) **A4_Scanned_Lab_Image.png** (scan)
   - Single page with a small lab table and messy alignment to test OCR.

## Patient B — Thyroid + Female Hormones
Goal: Multi-panel labs + imaging narrative + supplements.

Documents:
1) **B1_Thyroid_Hormone_Labs.pdf** (PDF, table-heavy)
   - Tests: TSH, Free T4, Free T3, TPO Ab, Estradiol, Progesterone, SHBG
   - Include ranges + flags
2) **B2_Thyroid_Ultrasound_Report.pdf** (PDF, narrative)
   - Findings: small nodules, benign impression
3) **B3_Supplements_List.txt** (text)
   - Supplements: selenium, iodine, vitamin D
4) **B4_Handwritten_Note_Scan.jpg** (scan)
   - Short note: fatigue, cold intolerance, hair loss

## Patient C — Inflammation + GI
Goal: Mixed labs, GI narrative, and OCR.

Documents:
1) **C1_Inflammation_Labs.pdf** (PDF, table-heavy)
   - Tests: CRP, ESR, Ferritin, Vitamin D, B12, Folate
2) **C2_GI_Clinic_Note.txt** (text)
   - Symptoms: bloating, constipation, food triggers
3) **C3_Stool_Test_Summary.pdf** (PDF, narrative + simple flags)
   - Findings: dysbiosis markers, low diversity
4) **C4_Scanned_Lab_Table.png** (scan)
   - Small table to test OCR + normalization.

## Patient D — Genetics / Pharmacogenomics
Goal: Genetics interpretation + medication response + labs.

Documents:
1) **D1_Genetic_Panel_Summary.pdf** (PDF, structured sections)
   - Mock variants: CYP2C19 *2/*2, MTHFR C677T (heterozygous)
   - Short interpretation blocks
2) **D2_Med_Response_Note.txt** (text)
   - “Poor metabolizer” note for clopidogrel
3) **D3_Liver_Coag_Labs.pdf** (PDF, table-heavy)
   - Tests: ALT, AST, INR, PT

## Format Mix and Feature Coverage
- OCR: A4_Scanned_Lab_Image.png, B4_Handwritten_Note_Scan.jpg, C4_Scanned_Lab_Table.png
- Table parsing + normalization: A1, B1, C1, D3
- Narrative extraction: A2, B2, C2, C3, D2
- Genetics: D1
- Meds: A3, D2
- RAG: all docs provide chunk diversity

## Folder Structure
Dataset is generated under `data/`:
- `data/patient_a/`
- `data/patient_b/`
- `data/patient_c/`
- `data/patient_d/`

## Notes for Generation
- Use synthetic names and dates.
- Keep units consistent (mg/dL, mmol/L, mIU/L, ng/mL).
- Add 1–2 abnormal flags per lab panel.
- For scans, keep text legible enough for OCR but with slight skew.

## Next Step
Bulk import via CLI:

```bash
python -m backend.scripts.import_mock_data
```

Use `--skip-embed` or `--skip-draft` if you want to avoid OpenAI calls.
