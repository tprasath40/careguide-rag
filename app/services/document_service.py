from io import BytesIO

from fastapi import HTTPException
from pypdf import PdfReader

from app.core.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be greater than or equal to zero "
            "and smaller than chunk size"
        )

    cleaned_text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        end = start + chunk_size
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def extract_text_from_file(filename: str, content: bytes) -> tuple[str, int]:

    lower_filename = filename.lower()

    if lower_filename.endswith(".txt"):
        return content.decode("utf-8"), 1

    if lower_filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(content))

        extracted_pages = [page.extract_text() or "" for page in reader.pages]

        return "\n".join(extracted_pages), len(reader.pages)

    raise HTTPException(
        status_code=415, detail="Only TXT and PDF files are currently supported"
    )
