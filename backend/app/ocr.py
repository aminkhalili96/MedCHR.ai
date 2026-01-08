from io import BytesIO
from typing import Optional

import pdfplumber
import pytesseract
from PIL import Image


def extract_text_from_pdf(data: bytes) -> str:
    text_parts = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def extract_text_from_image(data: bytes) -> str:
    image = Image.open(BytesIO(data))
    return pytesseract.image_to_string(image).strip()


def extract_text(data: bytes, content_type: str) -> str:
    if content_type == "application/pdf":
        return extract_text_from_pdf(data)
    if content_type.startswith("image/"):
        return extract_text_from_image(data)
    # Fallback: treat as plain text
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")
