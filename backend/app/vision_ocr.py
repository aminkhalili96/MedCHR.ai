"""
Vision-based OCR using GPT-4o Vision for complex medical documents.

This module handles:
- Table structure preservation
- Multi-column layouts
- Complex forms with checkboxes
- Handwritten annotations

Gap References: E01, E02, E09
"""

import base64
from io import BytesIO
import json

import pdfplumber

from .config import get_settings
from .llm_gateway import create_chat_completion


def extract_with_vision(data: bytes, content_type: str) -> dict:
    """
    Use GPT-4o Vision for layout-aware extraction.
    Returns structured text with table preservation.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return {"text": "", "method": "vision", "error": "No API key"}
    
    # Convert PDF pages to images if needed
    if content_type == "application/pdf":
        images = pdf_to_images(data)
    else:
        images = [data]
    
    all_text = []
    all_tables = []
    
    for idx, img_data in enumerate(images):
        result = extract_page_with_vision(img_data, page_num=idx + 1)
        all_text.append(result.get("text", ""))
        all_tables.extend(result.get("tables", []))
    
    return {
        "text": "\n\n---PAGE BREAK---\n\n".join(all_text),
        "tables": all_tables,
        "method": "vision",
        "pages": len(images)
    }


def pdf_to_images(data: bytes, dpi: int = 150) -> list[bytes]:
    """Convert PDF pages to images for vision processing."""
    images = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                # Convert page to image
                img = page.to_image(resolution=dpi)
                buf = BytesIO()
                img.save(buf, format="PNG")
                images.append(buf.getvalue())
    except Exception as e:
        print(f"PDF to image conversion failed: {e}")
    return images


def extract_page_with_vision(img_data: bytes, page_num: int = 1) -> dict:
    """Extract text and tables from a single page image."""
    
    b64_image = base64.b64encode(img_data).decode()
    
    # Detect content type
    if img_data[:8] == b'\x89PNG\r\n\x1a\n':
        mime_type = "image/png"
    elif img_data[:2] == b'\xff\xd8':
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"  # Default
    
    system_prompt = """You are a medical document OCR specialist. Extract ALL text from this medical document image.

CRITICAL RULES:
1. PRESERVE TABLE STRUCTURE - Use markdown table format with | delimiters
2. Maintain column alignment and headers
3. Extract ALL values including units and reference ranges
4. Note any checkboxes: [x] for checked, [ ] for unchecked
5. For handwritten text, transcribe as best as possible and mark with (handwritten)
6. Preserve dates exactly as shown

Return a JSON object:
{
    "text": "Full extracted text with tables in markdown format",
    "tables": [
        {
            "title": "Table title if present",
            "headers": ["Col1", "Col2", ...],
            "rows": [["val1", "val2", ...], ...]
        }
    ],
    "confidence": 0.0-1.0,
    "handwritten_sections": ["text that appears handwritten"]
}"""

    try:
        response = create_chat_completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Extract all content from page {page_num} of this medical document:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content) if content else {"text": "", "tables": []}
        
    except Exception as e:
        print(f"Vision extraction failed: {e}")
        return {"text": "", "tables": [], "error": str(e)}


def should_use_vision(data: bytes, content_type: str) -> bool:
    """
    Determine if vision-based OCR should be used.
    Uses vision for:
    - Images (always)
    - PDFs that likely contain tables or complex layouts
    """
    if content_type and content_type.startswith("image/"):
        return True
    
    if content_type == "application/pdf":
        # Check if PDF has tables using pdfplumber
        try:
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages[:3]:  # Check first 3 pages
                    tables = page.find_tables()
                    if tables:
                        return True
                    # Check for multi-column layout
                    if len(page.chars) > 100:
                        # Simple heuristic: if chars span wide x-range with gaps
                        xs = sorted(set(c['x0'] for c in page.chars))
                        if len(xs) > 50:  # Many distinct x positions
                            return True
        except:
            pass
    
    return False
