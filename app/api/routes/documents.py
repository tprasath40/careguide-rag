from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.document_service import chunk_text, extract_text_from_file
from app.services.retrieval_service import retrieval_service

router = APIRouter()


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:

    filename = file.filename or "uploaded-document"
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        text, pages = extract_text_from_file(filename, content)
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=422, detail="TXT file must use UTF-8 encoding"
        ) from error

    if not text.strip():
        raise HTTPException(
            status_code=422, detail="No readable text found in the document"
        )

    chunks = chunk_text(text)
    retrieval_service.store_document(filename, chunks)

    return {
        "filename": filename,
        "characters": len(text),
        "pages": pages,
        "chunks_created": len(chunks),
        "message": "Document indexed successfully",
    }
