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


def extract_text(data: bytes, content_type: str | None, use_vision: bool = None) -> str:
    """
    Extract text from document, optionally using Vision AI for complex layouts.
    
    Args:
        data: Raw file bytes
        content_type: MIME type
        use_vision: Force vision mode (None = auto-detect)
    """
    # Auto-detect if vision should be used
    if use_vision is None:
        try:
            from .vision_ocr import should_use_vision
            use_vision = should_use_vision(data, content_type)
        except ImportError:
            use_vision = False
    
    # Use Vision OCR for complex documents
    if use_vision:
        try:
            from .vision_ocr import extract_with_vision
            result = extract_with_vision(data, content_type)
            if result.get("text"):
                return result["text"]
        except Exception as e:
            print(f"Vision OCR failed, falling back to standard: {e}")
    
    # Standard extraction
    if content_type == "application/pdf":
        return extract_text_from_pdf(data)
    if content_type and content_type.startswith("image/"):
        return extract_text_from_image(data)
    # Fallback: treat as plain text
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")

