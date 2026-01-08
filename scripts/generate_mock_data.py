from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HOSPITAL_NAME = "Northwind Medical Center"
HOSPITAL_ADDRESS = "800 Northwind Ave, Harbor City, CA 94000"
HOSPITAL_PHONE = "(415) 555-0198"
HOSPITAL_CLIA = "CLIA: 00D0000000"
HOSPITAL_DIRECTOR = "Lab Director: Dr. Morgan Hale, MD"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_logo_commands(x: int = 40, y: int = 740) -> str:
    # Draw a simple cross logo using rectangles.
    return "\n".join(
        [
            "q",
            "0.09 0.45 0.85 rg",
            f"{x} {y + 10} 26 6 re f",
            f"{x + 10} {y} 6 26 re f",
            "Q",
        ]
    )


def pdf_text_stream(
    lines: list[str],
    start_x: int = 90,
    start_y: int = 760,
    font_size: int = 10,
    leading: int = 12,
) -> bytes:
    parts = [pdf_logo_commands(), "BT", f"/F1 {font_size} Tf", f"{leading} TL", f"{start_x} {start_y} Td"]
    if lines:
        parts.append(f"({escape_pdf_text(lines[0])}) Tj")
        for line in lines[1:]:
            parts.append("T*")
            parts.append(f"({escape_pdf_text(line)}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def write_simple_pdf(path: Path, lines: list[str], font_size: int = 10, leading: int = 12) -> None:
    content = pdf_text_stream(lines, font_size=font_size, leading=leading)
    obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
    obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    obj3 = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    obj4 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    obj5 = b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"

    objects = [obj1, obj2, obj3, obj4, obj5]
    output = bytearray()
    output.extend(b"%PDF-1.4\n")

    offsets = [0]
    for i, data in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{i} 0 obj\n".encode("ascii"))
        output.extend(data)
        if not data.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(b"endobj\n")

    xref_start = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    output.extend(b"trailer\n")
    output.extend(f"<< /Size {len(objects)+1} /Root 1 0 R >>\n".encode("ascii"))
    output.extend(b"startxref\n")
    output.extend(f"{xref_start}\n".encode("ascii"))
    output.extend(b"%%EOF\n")

    path.write_bytes(output)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_logo(draw: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    color = (24, 90, 189)
    draw.ellipse([x, y, x + size, y + size], outline=color, width=3)
    bar = int(size * 0.22)
    center = x + size // 2
    draw.rectangle([center - bar // 2, y + int(size * 0.2), center + bar // 2, y + int(size * 0.8)], fill=color)
    draw.rectangle([x + int(size * 0.2), y + size // 2 - bar // 2, x + int(size * 0.8), y + size // 2 + bar // 2], fill=color)


def write_text_image(path: Path, lines: list[str], size=(1400, 1100), font_size=20, rotate=0) -> None:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)
    draw_logo(draw, 40, 30, 60)
    y = 30
    x = 120
    for idx, line in enumerate(lines):
        draw.text((x, y), line, fill="black", font=font)
        y += font_size + (8 if idx < 4 else 6)
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor="white")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(path, format="JPEG", quality=90)
    else:
        img.save(path, format="PNG")


def header_lines(
    patient: str,
    dob: str,
    report_id: str,
    collection_date: str,
    sex: str,
    specimen: str = "Serum",
) -> list[str]:
    mrn = report_id.replace("-", "")[-8:]
    return [
        f"{HOSPITAL_NAME} - Clinical Laboratory Services",
        f"{HOSPITAL_ADDRESS} | {HOSPITAL_PHONE}",
        f"{HOSPITAL_CLIA} | {HOSPITAL_DIRECTOR}",
        "Ordering Provider: Dr. Lena Park | Location: Internal Medicine",
        f"Patient: {patient} | DOB: {dob} | Sex: {sex} | MRN: {mrn}",
        f"Report ID: {report_id} | Accession: {report_id}",
        f"Collected: {collection_date} 08:14 | Received: {collection_date} 11:03 | Specimen: {specimen}",
        "Report Status: Final | Fasting: Yes | Method: Automated Analyzer",
        "------------------------------------------------------------------",
    ]


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines))


def main() -> None:
    patients = {
        "patient_a": {"name": "Alex Parker", "dob": "1982-05-14", "sex": "M"},
        "patient_b": {"name": "Bianca Flores", "dob": "1990-11-02", "sex": "F"},
        "patient_c": {"name": "Chris Nordin", "dob": "1978-03-09", "sex": "M"},
        "patient_d": {"name": "Dana Lee", "dob": "1986-12-22", "sex": "F"},
    }
    for folder in patients:
        ensure_dir(DATA / folder)

    # Patient A
    patient_a = patients["patient_a"]
    write_simple_pdf(
        DATA / "patient_a" / "A1_Labs_Metabolic.pdf",
        header_lines(patient_a["name"], patient_a["dob"], "NWMC-2024-0612-A", "2024-06-12", patient_a["sex"])
        + [
            "Report Type: Comprehensive Metabolic & Cardio Panel",
            "Panel: Lipid Panel (fasting)",
            "Test | Value | Unit | Range | Flag",
            "Total Cholesterol | 248 | mg/dL | 125-200 | H",
            "HDL | 38 | mg/dL | 40-60 | L",
            "LDL (calculated) | 162 | mg/dL | 0-129 | H",
            "Non-HDL Cholesterol | 210 | mg/dL | 0-159 | H",
            "Triglycerides | 190 | mg/dL | 0-150 | H",
            "Apolipoprotein B | 126 | mg/dL | 60-120 | H",
            "Lipoprotein(a) | 58 | mg/dL | 0-30 | H",
            "Panel: Glucose Control",
            "Glucose | 118 | mg/dL | 70-99 | H",
            "HbA1c | 6.1 | % | 4.0-5.6 | H",
            "Estimated Avg Glucose | 128 | mg/dL | 70-117 | H",
            "Insulin (fasting) | 18.4 | uIU/mL | 2.6-24.9 | N",
            "Panel: Comprehensive Metabolic Panel",
            "Sodium | 139 | mmol/L | 134-144 | N",
            "Potassium | 4.6 | mmol/L | 3.5-5.2 | N",
            "Chloride | 101 | mmol/L | 96-106 | N",
            "CO2 | 24 | mmol/L | 20-29 | N",
            "BUN | 16 | mg/dL | 6-24 | N",
            "Creatinine | 1.0 | mg/dL | 0.6-1.3 | N",
            "eGFR | 88 | mL/min/1.73m2 | >60 | N",
            "Calcium | 9.4 | mg/dL | 8.6-10.2 | N",
            "Total Protein | 7.2 | g/dL | 6.0-8.5 | N",
            "Albumin | 4.5 | g/dL | 3.8-5.0 | N",
            "Globulin | 2.7 | g/dL | 1.5-4.5 | N",
            "A/G Ratio | 1.7 | ratio | 1.1-2.5 | N",
            "Total Bilirubin | 0.8 | mg/dL | 0.0-1.2 | N",
            "Alkaline Phosphatase | 102 | U/L | 44-121 | N",
            "AST | 44 | U/L | 10-40 | H",
            "ALT | 52 | U/L | 7-56 | N",
            "Panel: CBC",
            "WBC | 7.4 | x10^3/uL | 3.4-10.8 | N",
            "RBC | 5.02 | x10^6/uL | 4.14-5.80 | N",
            "Hemoglobin | 14.8 | g/dL | 13.0-17.7 | N",
            "Hematocrit | 44.2 | % | 37.5-51.0 | N",
            "MCV | 88 | fL | 79-97 | N",
            "MCH | 29.5 | pg | 26.6-33.0 | N",
            "RDW | 13.1 | % | 11.6-15.4 | N",
            "Platelets | 252 | x10^3/uL | 150-450 | N",
            "hs-CRP | 3.4 | mg/L | 0.0-3.0 | H",
            "Interpretive Comment: Pattern suggests metabolic syndrome risk; correlate clinically.",
            "Notes: Synthetic demo data",
        ],
    )

    write_text(
        DATA / "patient_a" / "A2_Progress_Note.txt",
        [
            f"{HOSPITAL_NAME} - Internal Medicine Progress Note",
            "Patient: Alex Parker | DOB: 1982-05-14 | Visit Date: 2024-06-20",
            "Chief Complaint: fatigue, weight gain, reduced exercise tolerance",
            "HPI: 6-month history of low energy and gradual weight gain (~12 lb).",
            "Diet high in processed carbs; sleep 6 hrs/night; sedentary job.",
            "ROS: denies chest pain, syncope; occasional heartburn; no polyuria.",
            "PMH: elevated lipids noted in 2022. No known CAD.",
            "Medications: see medication list. Allergies: NKDA.",
            "Vitals: BP 138/86, HR 78, BMI 29.4.",
            "Assessment: dyslipidemia, prediabetes range A1c, metabolic syndrome risk.",
            "Plan: lifestyle counseling, exercise 150 min/week, recheck labs in 12 weeks.",
            "Follow-up: consider statin intensification if LDL remains elevated.",
            "Note: Synthetic demo data",
        ],
    )

    write_simple_pdf(
        DATA / "patient_a" / "A3_Meds_List.pdf",
        header_lines(patient_a["name"], patient_a["dob"], "NWMC-2024-0612-M", "2024-06-12", patient_a["sex"])
        + [
            "Medication List",
            "Metformin 500 mg, PO, BID | Start: 2023-11-15 | Prescriber: Dr. Park",
            "Atorvastatin 20 mg, PO, nightly | Start: 2022-09-03 | Prescriber: Dr. Park",
            "Vitamin D3 2000 IU daily | OTC",
            "Fish Oil 1000 mg daily | OTC",
            "Allergies: NKDA",
            "Notes: Synthetic demo data",
        ],
    )

    write_text_image(
        DATA / "patient_a" / "A4_Scanned_Lab_Image.png",
        header_lines(patient_a["name"], patient_a["dob"], "NWMC-2024-0612-S", "2024-06-12", patient_a["sex"])
        + [
            "Lab Summary (scan)",
            "LDL 162 mg/dL (H)",
            "HDL 38 mg/dL (L)",
            "Non-HDL 210 mg/dL (H)",
            "HbA1c 6.1 % (H)",
            "Glucose 118 mg/dL (H)",
            "hs-CRP 3.4 mg/L (H)",
            "Notes: Synthetic demo data",
        ],
        rotate=1,
    )

    # Patient B
    patient_b = patients["patient_b"]
    write_simple_pdf(
        DATA / "patient_b" / "B1_Thyroid_Hormone_Labs.pdf",
        header_lines(patient_b["name"], patient_b["dob"], "NWMC-2024-0701-B", "2024-07-01", patient_b["sex"])
        + [
            "Report Type: Thyroid + Hormone Panel",
            "Panel: Thyroid Function",
            "Test | Value | Unit | Range | Flag",
            "TSH | 6.2 | mIU/L | 0.4-4.0 | H",
            "Free T4 | 0.7 | ng/dL | 0.8-1.8 | L",
            "Total T4 | 4.5 | ug/dL | 4.5-12.0 | N",
            "Free T3 | 2.1 | pg/mL | 2.3-4.2 | L",
            "Reverse T3 | 24 | ng/dL | 9-24 | N",
            "TPO Ab | 128 | IU/mL | 0-34 | H",
            "Thyroglobulin Ab | 5.6 | IU/mL | 0-0.9 | H",
            "TSI | 0.3 | IU/L | 0.0-0.55 | N",
            "Panel: Female Hormones (Cycle Day 3)",
            "FSH | 7.2 | mIU/mL | 3.5-12.5 | N",
            "LH | 6.1 | mIU/mL | 2.4-12.6 | N",
            "Estradiol | 52 | pg/mL | 30-400 | N",
            "Progesterone | 0.6 | ng/mL | 0.1-0.8 | N",
            "Total Testosterone | 24 | ng/dL | 8-60 | N",
            "DHEA-S | 210 | ug/dL | 65-380 | N",
            "Prolactin | 22.4 | ng/mL | 4.8-23.3 | N",
            "AM Cortisol | 17.8 | ug/dL | 6.2-19.4 | N",
            "AMH | 1.8 | ng/mL | 0.5-4.0 | N",
            "Interpretive Comment: Thyroid pattern consistent with hypothyroidism; correlate with symptoms.",
            "Notes: Synthetic demo data",
        ],
    )

    write_simple_pdf(
        DATA / "patient_b" / "B2_Thyroid_Ultrasound_Report.pdf",
        header_lines(patient_b["name"], patient_b["dob"], "NWMC-2024-0703-U", "2024-07-03", patient_b["sex"])
        + [
            "Thyroid Ultrasound Report",
            "Technique: High-resolution ultrasound with grayscale and Doppler.",
            "Right lobe: 4.2 x 1.4 x 1.3 cm, homogeneous echotexture.",
            "Left lobe: 3.9 x 1.3 x 1.2 cm, mildly heterogeneous.",
            "Isthmus thickness: 0.3 cm.",
            "Nodules: Right lobe 6 mm hypoechoic nodule, TI-RADS 2.",
            "No suspicious calcifications or irregular margins.",
            "Vascularity: normal.",
            "Cervical lymph nodes: no abnormal nodes detected.",
            "Impression: benign-appearing nodules. Recommend routine follow-up.",
            "Note: Synthetic demo data",
        ],
    )

    write_text(
        DATA / "patient_b" / "B3_Supplements_List.txt",
        [
            f"{HOSPITAL_NAME} - Supplements List",
            "Patient: Bianca Flores | DOB: 1990-11-02",
            "Supplements:",
            "- Selenium 200 mcg daily",
            "- Iodine 150 mcg daily",
            "- Vitamin D3 2000 IU daily",
            "- Magnesium glycinate 200 mg nightly",
            "- Omega-3 1000 mg daily",
            "Note: Synthetic demo data",
        ],
    )

    write_text_image(
        DATA / "patient_b" / "B4_Handwritten_Note_Scan.jpg",
        header_lines(patient_b["name"], patient_b["dob"], "NWMC-2024-0704-H", "2024-07-04", patient_b["sex"])
        + [
            "Note (scan)",
            "fatigue, cold intolerance",
            "hair loss, dry skin",
            "constipation, brain fog",
            "sleep 7 hrs, low energy",
            "Notes: Synthetic demo data",
        ],
        rotate=-2,
    )

    # Patient C
    patient_c = patients["patient_c"]
    write_simple_pdf(
        DATA / "patient_c" / "C1_Inflammation_Labs.pdf",
        header_lines(patient_c["name"], patient_c["dob"], "NWMC-2024-0628-C", "2024-06-28", patient_c["sex"])
        + [
            "Report Type: Inflammation + Iron Studies",
            "Panel: Inflammatory Markers",
            "Test | Value | Unit | Range | Flag",
            "CRP | 6.2 | mg/L | 0-3 | H",
            "hs-CRP | 4.1 | mg/L | 0.0-3.0 | H",
            "ESR | 28 | mm/hr | 0-20 | H",
            "IL-6 | 6.8 | pg/mL | 0.0-7.0 | N",
            "Panel: Iron Studies",
            "Ferritin | 22 | ng/mL | 30-300 | L",
            "Serum Iron | 52 | ug/dL | 38-169 | N",
            "TIBC | 390 | ug/dL | 250-450 | N",
            "Iron Saturation | 13 | % | 15-55 | L",
            "Panel: Vitamins",
            "Vitamin D | 22 | ng/mL | 30-100 | L",
            "B12 | 310 | pg/mL | 200-900 | N",
            "Folate | 5.2 | ng/mL | 3.0-17.0 | N",
            "Panel: Autoimmune Screening",
            "ANA | Negative | titer | Negative | N",
            "RF | 12 | IU/mL | 0-14 | N",
            "CCP IgG | 7 | U/mL | 0-19 | N",
            "Interpretive Comment: Elevated CRP/ESR suggest inflammation; iron stores low.",
            "Notes: Synthetic demo data",
        ],
    )

    write_text(
        DATA / "patient_c" / "C2_GI_Clinic_Note.txt",
        [
            f"{HOSPITAL_NAME} - GI Clinic Note",
            "Patient: Chris Nordin | DOB: 1978-03-09 | Visit Date: 2024-06-30",
            "Chief Complaint: bloating, constipation, intermittent abdominal discomfort.",
            "HPI: Symptoms for 8 months, worsened by dairy and high FODMAP foods.",
            "ROS: no hematochezia, occasional nausea, no weight loss.",
            "PMH: seasonal allergies, no prior GI procedures.",
            "Medications: none; Allergies: NKDA.",
            "Assessment: IBS vs. low-grade inflammatory process.",
            "Plan: elimination diet trial, fiber increase, stool panel ordered.",
            "Note: Synthetic demo data",
        ],
    )

    write_simple_pdf(
        DATA / "patient_c" / "C3_Stool_Test_Summary.pdf",
        header_lines(patient_c["name"], patient_c["dob"], "NWMC-2024-0702-S", "2024-07-02", patient_c["sex"], specimen="Stool")
        + [
            "Stool Test Summary",
            "Marker | Value | Unit | Range | Flag",
            "Calprotectin | 90 | ug/g | <50 | H",
            "Lactoferrin | 10 | ug/g | <7.2 | H",
            "Pancreatic Elastase | 320 | ug/g | >200 | N",
            "Secretory IgA | 320 | mg/dL | 51-204 | H",
            "Occult Blood | Negative | result | Negative | N",
            "Parasites | Not detected | result | Negative | N",
            "H. pylori | Not detected | result | Negative | N",
            "Impression: mild inflammatory pattern; correlate with symptoms.",
            "Notes: Synthetic demo data",
        ],
    )

    write_text_image(
        DATA / "patient_c" / "C4_Scanned_Lab_Table.png",
        header_lines(patient_c["name"], patient_c["dob"], "NWMC-2024-0628-S", "2024-06-28", patient_c["sex"])
        + [
            "Lab Table (scan)",
            "CRP 6.2 mg/L (H)",
            "hs-CRP 4.1 mg/L (H)",
            "ESR 28 mm/hr (H)",
            "Ferritin 22 ng/mL (L)",
            "Iron Saturation 13 % (L)",
            "Vitamin D 22 ng/mL (L)",
            "Notes: Synthetic demo data",
        ],
        rotate=1,
    )

    # Patient D
    patient_d = patients["patient_d"]
    write_simple_pdf(
        DATA / "patient_d" / "D1_Genetic_Panel_Summary.pdf",
        header_lines(patient_d["name"], patient_d["dob"], "NWMC-2024-0705-G", "2024-07-05", patient_d["sex"], specimen="Saliva")
        + [
            "Genetic Panel Summary (Pharmacogenomics + Risk Markers)",
            "Genes tested: CYP2C19, CYP2D6, SLCO1B1, VKORC1, MTHFR, APOE",
            "Result | Genotype | Interpretation",
            "CYP2C19 | *2/*2 | Poor metabolizer; reduced response to clopidogrel",
            "CYP2D6 | *1/*4 | Intermediate metabolizer; consider dose adjustments",
            "SLCO1B1 | *5/*1 | Increased statin myopathy risk",
            "VKORC1 | -1639 A/G | Typical warfarin sensitivity",
            "MTHFR | C677T (heterozygous) | Mild reduction in enzyme activity",
            "APOE | e3/e3 | Typical population risk",
            "Interpretive Comment: Review med list for CYP2C19 and SLCO1B1 interactions.",
            "Limitations: This test does not assess all genetic variants.",
            "Notes: Synthetic demo data",
        ],
    )

    write_text(
        DATA / "patient_d" / "D2_Med_Response_Note.txt",
        [
            f"{HOSPITAL_NAME} - Medication Response Note",
            "Patient: Dana Lee | DOB: 1986-12-22",
            "CYP2C19 poor metabolizer; avoid clopidogrel if alternatives available.",
            "SLCO1B1 variant suggests higher statin myopathy risk; monitor symptoms.",
            "Consider prasugrel or ticagrelor if clinically indicated.",
            "Note: Synthetic demo data",
        ],
    )

    write_simple_pdf(
        DATA / "patient_d" / "D3_Liver_Coag_Labs.pdf",
        header_lines(patient_d["name"], patient_d["dob"], "NWMC-2024-0705-L", "2024-07-05", patient_d["sex"])
        + [
            "Panel: Liver and Coagulation",
            "Test | Value | Unit | Range | Flag",
            "ALT | 68 | U/L | 7-56 | H",
            "AST | 55 | U/L | 10-40 | H",
            "GGT | 72 | U/L | 9-48 | H",
            "Alkaline Phosphatase | 128 | U/L | 44-121 | H",
            "Total Bilirubin | 1.4 | mg/dL | 0.0-1.2 | H",
            "Albumin | 4.1 | g/dL | 3.8-5.0 | N",
            "INR | 1.3 | ratio | 0.9-1.1 | H",
            "PT | 14.5 | sec | 11.0-13.5 | H",
            "aPTT | 34 | sec | 25-35 | N",
            "Interpretive Comment: Mild transaminitis; correlate with medications.",
            "Notes: Synthetic demo data",
        ],
    )

    print(f"Mock data written to: {DATA}")


if __name__ == "__main__":
    main()
