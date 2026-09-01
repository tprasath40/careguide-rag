import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services.document_service import chunk_text, extract_text_from_file


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "careguide-rag"
    }


def test_chunk_text_creates_overlapping_chunks():
    text = "A" * 1000

    chunks = chunk_text(
        text=text,
        chunk_size=500,
        overlap=100
    )

    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            text="sample text",
            chunk_size=100,
            overlap=100
        )


def test_extract_text_file():
    content = b"Healthcare document content"

    text, pages = extract_text_from_file(
        "guide.txt",
        content
    )

    assert text == "Healthcare document content"
    assert pages == 1


def test_rejects_unsupported_file():
    with pytest.raises(HTTPException) as error:
        extract_text_from_file(
            "image.png",
            b"sample"
        )

    assert error.value.status_code == 415
