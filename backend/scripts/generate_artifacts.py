
import os
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DATA_DIR = Path("data")

# Full 14-Patient List matching the new complex profiles in seed_data.py
PATIENTS = [
    # --- The Restored 4 ---
    {
        "name": "Arthur_Morgan", "condition": "COPD (GOLD 4)",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["FEV1", "0.90 L", "> 3.10", "LOW"], ["FVC", "2.10 L", "> 4.05", "LOW"], ["FEV1/FVC", "0.42", "> 0.70", "LOW"]],
        "notes": "Severe irreversible obstruction. Diffusion capacity severely reduced."
    },
    {
        "name": "Beatrice_Kiddo", "condition": "Sickle Cell Disease",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["Hemoglobin", "7.2 g/dL", "12.0-16.0", "LOW"], ["Reticulocytes", "12.5%", "0.5-2.5", "HIGH"], ["Total Bilirubin", "2.5 mg/dL", "0.1-1.2", "HIGH"]],
        "notes": "Peripheral smear shows frequent sickle cells. Consistent with hemolytic anemia."
    },
    {
        "name": "Charles_Xavier", "condition": "ESRD (Hemodialysis)",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["Creatinine", "8.4 mg/dL", "0.6-1.2", "HIGH"], ["Potassium", "5.6 mmol/L", "3.5-5.1", "HIGH"], ["Phosphorus", "6.8 mg/dL", "2.5-4.5", "HIGH"]],
        "notes": "Pre-dialysis sample. Hyperkalemia and Hyperphosphatemia present."
    },
    {
        "name": "Diana_Prince", "condition": "Psoriatic Arthritis",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["CRP", "24.5 mg/L", "< 10.0", "HIGH"], ["ESR", "48 mm/hr", "0-20", "HIGH"], ["RF", "Negative", "< 14", ""]],
        "notes": "Elevated inflammatory markers consistent with active PsA flare."
    },
    # --- The Enhanced 10 ---
    {
        "name": "John_Doe", "condition": "Cardiology/CKD",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["Troponin I", "0.04 ng/mL", "< 0.03", "HIGH"], ["BNP", "450 pg/mL", "< 100", "HIGH"], ["Creatinine", "1.5 mg/dL", "0.7-1.3", "HIGH"]],
        "notes": "Mild Troponin leak. Elevated BNP suggests fluid overload."
    },
    {
        "name": "Maria_Rodriguez", "condition": "Ankylosing Spondylitis",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["HLA-B27", "POSITIVE", "Negative", "ABNORMAL"], ["CRP", "15.0 mg/L", "< 10.0", "HIGH"]],
        "notes": "HLA-B27 Positive supporting clinical diagnosis of AS."
    },
    {
        "name": "Robert_Chen", "condition": "NSCLC (Lung Cancer)",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["CEA", "15.2 ng/mL", "< 3.0", "HIGH"], ["EGFR Mutation", "Exon 19 Del", "None", "DETECTED"]],
        "notes": "Molecular profiling confirms EGFR Exon 19 Deletion."
    },
    {
        "name": "Emily_Blunt", "condition": "Multiple Sclerosis",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["CSF Oligoclonal Bands", "POSITIVE", "Negative", "ABNORMAL"], ["IgG Index", "0.85", "< 0.7", "HIGH"]],
        "notes": "CSF findings supportive of demyelinating process."
    },
    {
        "name": "Michael_Johnson", "condition": "Type 1 Diabetes",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["HbA1c", "6.9%", "< 5.7", "HIGH"], ["Glucose (Random)", "65 mg/dL", "70-140", "LOW"]],
        "notes": "HbA1c indicates reasonable control, but random glucose shows hypoglycemia."
    },
    {
        "name": "Sarah_Connor", "condition": "PTSD / Depression",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["CYP2D6 Genotype", "*4/*4", "Normal", "POOR METABOLIZER"], ["TSH", "1.4 uIU/mL", "0.4-4.0", "Normal"]],
        "notes": "Pharmacogenetic testing indicates Poor Metabolizer status for CYP2D6."
    },
    {
        "name": "David_Kim", "condition": "HIV / Hep B",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["HIV-1 RNA", "< 20 copies", "-", "UNDETECTABLE"], ["CD4 Count", "850 cells/uL", "500-1500", "Normal"]],
        "notes": "Virologic suppression achieved. Immune status preserved."
    },
    {
        "name": "Linda_Hamilton", "condition": "Alzheimer's Dementia",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["Vitamin B12", "350 pg/mL", "200-900", "Normal"], ["TSH", "2.1 uIU/mL", "0.4-4.0", "Normal"], ["RPR", "Non-Reactive", "Non-Reactive", ""]],
        "notes": "Reversible causes of dementia (Vitamin def, Hypothyroid, Syphilis) ruled out."
    },
    {
        "name": "James_Bond", "condition": "Trauma (Post-Splenectomy)",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["Platelets", "550 K/uL", "150-400", "HIGH"], ["WBC", "12.5 K/uL", "4.5-11.0", "HIGH"], ["Hemoglobin", "10.0 g/dL", "13.0-17.0", "LOW"]],
        "notes": "Thrombocytosis and Leukocytosis consistent with post-splenectomy state."
    },
    {
        "name": "Fiona_Gallagher",
        "condition": "Crohn's Disease",
        "labs": [["Test", "Result", "Ref Range", "Flag"], ["Fecal Calprotectin", "800 ug/g", "< 50", "HIGH"], ["Albumin", "3.2 g/dL", "3.5-5.0", "LOW"], ["Iron", "40 ug/dL", "60-170", "LOW"]],
        "notes": "Significantly elevated Calprotectin indicates active bowel inflammation."
    }
]

def create_pdf(patient):
    folder = DATA_DIR / patient["name"]
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / "lab_result.pdf"
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Header
    elements.append(Paragraph("<b>ST. MARY'S REGIONAL MEDICAL CENTER</b>", styles['Title']))
    elements.append(Paragraph("Department of Pathology and Laboratory Medicine", styles['Heading4']))
    elements.append(Spacer(1, 20))
    
    # Patient Info
    info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=12, leading=14)
    elements.append(Paragraph(f"<b>Patient:</b> {patient['name'].replace('_', ' ')}", info_style))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}", info_style))
    elements.append(Paragraph(f"<b>Condition/Indication:</b> {patient['condition']}", info_style))
    elements.append(Spacer(1, 20))
    
    # Table
    table_data = patient["labs"]
    # Add colors to flags
    processed_data = []
    # Header row
    processed_data.append(table_data[0]) 
    
    # Data rows
    row_styles = []
    for i, row in enumerate(table_data[1:], start=1):
        processed_data.append(row)
        flag = row[3]
        if "HIGH" in flag or "LOW" in flag or "ABNORMAL" in flag or "DETECTED" in flag or "POOR" in flag:
            # Row index i (account for header at 0)
            row_styles.append(('TEXTCOLOR', (3, i), (3, i), colors.red))
            row_styles.append(('FONTNAME', (3, i), (3, i), 'Helvetica-Bold'))

    t = Table(processed_data)
    
    base_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]
    t.setStyle(TableStyle(base_style + row_styles))
    
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # Notes
    elements.append(Paragraph(f"<b>Pathologist Interpretation:</b> {patient['notes']}", styles['Normal']))
    elements.append(Spacer(1, 40))
    
    # Footer
    elements.append(Paragraph("Electronically Signed by: Dr. A. Chen, MD", styles['Italic']))
    
    doc.build(elements)
    print(f"Generated PDF: {pdf_path}")

def rename_images():
    # Helper to standardize any 'Scanned' images to 'data.png' if they exist
    for p in PATIENTS:
        folder = DATA_DIR / p["name"]
        src = folder / "Scanned_Lab_Image.png"
        dst = folder / "data.png"
        if src.exists():
            src.rename(dst)
            print(f"Renamed {src} to {dst}")

if __name__ == "__main__":
    print(f"Starting Artifact Generation for {len(PATIENTS)} patients...")
    for p in PATIENTS:
        try:
            create_pdf(p)
        except Exception as e:
            print(f"Failed to generate PDF for {p['name']}: {e}")
    rename_images()
    print("Artifact generation complete.")
