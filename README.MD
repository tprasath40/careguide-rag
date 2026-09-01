# CareGuide RAG API

CareGuide is a healthcare document question-answering API built with
Python, FastAPI and Claude.

Users can upload PDF or TXT documents and ask questions about their
contents. The application retrieves relevant passages and instructs
Claude to answer only from that context with source citations.

> CareGuide provides document-based information only and is not a
> substitute for professional medical advice.

## Features

- PDF and TXT document upload
- Text extraction and validation
- Configurable overlapping text chunking
- TF-IDF vectorization
- Cosine-similarity retrieval
- Configurable top-k context selection
- Claude-powered grounded answers
- Document and chunk citations
- Healthcare safety prompt
- Interactive Swagger API documentation
- Automated tests

## RAG Pipeline

```text
Document upload
    ↓
PDF/TXT text extraction
    ↓
Text cleaning and overlapping chunking
    ↓
TF-IDF vectorization
    ↓
Cosine-similarity top-k retrieval
    ↓
Grounded prompt construction
    ↓
Claude answer with source citations


## Technology Stack

Python
FastAPI
Anthropic Claude API
scikit-learn
PyPDF
Pydantic
Pytest

Project Structure

careguide-rag/
├── app/
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt


Setup

Create and activate a virtual environment:

python -m venv .venv

Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Create .env from .env.example and provide an Anthropic API key:

ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=claude-haiku-4-5

Run the application:

uvicorn app.main:app --reload

Swagger documentation:

http://127.0.0.1:8000/docs
API Endpoints
Health check
GET /health
Upload and index a document
POST /documents/upload

Supported formats:

PDF
TXT
Ask a question
POST /query

Example request:

{
  "question": "What information does the document provide?",
  "top_k": 3
}

The response includes the grounded answer, retrieved source chunks,
similarity scores and a healthcare disclaimer.

Design Decisions
Why chunk overlap?

Overlap helps preserve information that crosses chunk boundaries. The
current default uses 500-character chunks with a 100-character overlap.

Why TF-IDF for the MVP?

TF-IDF provides lightweight and explainable local retrieval without
requiring a large model download or external vector database.

A future iteration will introduce dense semantic embeddings and a
persistent vector store while retaining TF-IDF for hybrid retrieval.

Why citations?

Returning filename and chunk identifiers makes answers traceable and
helps users verify whether the response is grounded in the document.

Current Limitations
Document vectors are stored in memory and disappear after restart.
Scanned PDFs require OCR and are not currently supported.
TF-IDF is based on lexical similarity rather than semantic similarity.
The MVP runs as a single application process.
Authentication and tenant isolation are not yet implemented.
Roadmap
Dense semantic embeddings
Persistent vector database
Hybrid retrieval and reranking
Metadata filtering
Retrieval-quality evaluation
Prompt-injection protection
Authentication and tenant isolation
Asynchronous document ingestion
Streaming responses
Docker and cloud deployment
Tests
pytest -q
Author

G Thiruvarangan Prasath

Senior Full-Stack / Backend Engineer and Technical Lead with experience
in Node.js, TypeScript, React, Python, healthcare platforms and
AI-enabled applications.