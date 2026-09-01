# CareGuide RAG API

CareGuide is a modular healthcare document question-answering API built with Python, FastAPI, LangGraph and Anthropic Claude.

Users can upload PDF or TXT documents and ask questions about their contents. The application retrieves relevant passages and asks Claude to answer only from the supplied context with document and chunk citations.

> **Disclaimer:** CareGuide provides document-based information only and is not a substitute for professional medical advice.

## Features

- PDF and TXT document ingestion
- Text extraction and validation
- Configurable overlapping text chunking
- TF-IDF document vectorization
- Cosine-similarity top-k retrieval
- LangGraph workflow orchestration
- Conditional generate/fallback routing
- Claude-powered grounded answers
- Document and chunk citations
- Real-time Claude response streaming
- Server-Sent Events with structured event types
- Healthcare safety guardrails
- Relevance-threshold fallback
- Modular service-oriented Python architecture
- Interactive Swagger API documentation
- Automated API, workflow and streaming tests

## Application Flows

### Standard RAG query

```text
Document upload
    ↓
PDF/TXT text extraction
    ↓
Text cleaning and overlapping chunking
    ↓
TF-IDF vectorization and in-memory indexing
    ↓
POST /query
    ↓
LangGraph retrieve node
    ↓
Relevance decision
    ├── Relevant context
    │       ↓
    │   Claude grounded generation
    │       ↓
    │   JSON answer with citations
    │
    └── Insufficient context
            ↓
        Deterministic fallback response
```

### Streaming RAG query

```text
POST /query/stream
    ↓
Top-k document retrieval
    ↓
Relevance check
    ├── Relevant context
    │       ↓
    │   Claude streaming API
    │       ↓
    │   SSE sources → tokens → done
    │
    └── Insufficient context
            ↓
        SSE sources → fallback → done
```

## Technology Stack

- **Language:** Python
- **API framework:** FastAPI
- **Workflow orchestration:** LangGraph
- **LLM:** Anthropic Claude
- **Retrieval:** TF-IDF and cosine similarity
- **Document processing:** PyPDF
- **Validation:** Pydantic
- **Streaming:** Server-Sent Events
- **Testing:** Pytest and FastAPI TestClient

## Project Structure

```text
careguide-rag/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── documents.py
│   │       └── query.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   ├── retrieval_service.py
│   │   ├── claude_service.py
│   │   └── streaming_service.py
│   └── workflows/
│       ├── __init__.py
│       └── rag_workflow.py
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_rag_workflow.py
│   └── test_streaming_service.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Architecture

The application is divided into focused layers:

| Layer | Responsibility |
|---|---|
| API routes | HTTP request handling and response construction |
| Models | Pydantic request validation and LangGraph state |
| Document service | PDF/TXT extraction, cleaning and chunking |
| Retrieval service | In-memory indexing, TF-IDF vectorization and top-k retrieval |
| Claude service | Grounded synchronous and streamed generation |
| Streaming service | SSE event formatting and stream coordination |
| RAG workflow | LangGraph nodes, conditional routing and graph compilation |
| Core configuration | Environment access, constants, prompts and safety messages |

Modules follow a one-directional dependency structure:

```text
main → API routes → workflows/services → models/core
```

A single `RetrievalService` instance owns the in-memory index and is shared across document upload, normal query and streaming query routes.

This ensures all endpoints operate on the same indexed documents while avoiding circular imports and duplicated retrieval state.

## LangGraph Orchestration

The standard `/query` endpoint invokes a compiled LangGraph workflow with shared typed state:

```text
START → retrieve → relevance decision
                     ├── generate → END
                     └── fallback → END
```

The graph state contains:

- User question
- Requested `top_k`
- Retrieved document chunks
- Generated or fallback answer

The retrieval node selects the highest-ranking document chunks. A conditional edge compares the highest similarity score against the configured relevance threshold.

Relevant context is sent to Claude for grounded generation. Insufficient context is routed to a deterministic fallback without making an unnecessary LLM request.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tprasath40/careguide-rag.git
cd careguide-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Claude

Create a `.env` file using `.env.example` as the reference:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=claude-haiku-4-5
```

Never commit the real `.env` file or API key.

### 5. Run the application

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "careguide-rag"
}
```

### Upload and index a document

```http
POST /documents/upload
```

Supported formats:

- PDF
- TXT

The endpoint:

1. Validates the uploaded document.
2. Extracts readable text.
3. Cleans and splits the text into overlapping chunks.
4. Builds TF-IDF vectors.
5. Stores the chunks and vectors in memory.

Example response:

```json
{
  "filename": "healthcare-guide.pdf",
  "characters": 6200,
  "pages": 3,
  "chunks_created": 16,
  "message": "Document indexed successfully"
}
```

### Ask a question

```http
POST /query
```

Example request:

```json
{
  "question": "What information does the document provide?",
  "top_k": 3
}
```

Example response structure:

```json
{
  "question": "What information does the document provide?",
  "answer": "Grounded answer with document citations.",
  "sources": [
    {
      "filename": "healthcare-guide.pdf",
      "chunk_id": 2,
      "score": 0.1845,
      "preview": "Retrieved source preview..."
    }
  ],
  "workflow": [
    "retrieve",
    "generate"
  ],
  "disclaimer": "This application provides document-based information only and is not a substitute for professional medical advice."
}
```

### Stream a grounded answer

```http
POST /query/stream
```

Request:

```json
{
  "question": "What information does the document provide?",
  "top_k": 3
}
```

The response uses the `text/event-stream` media type.

Successful event sequence:

```text
event: sources
data: {"sources": [...]}

event: token
data: {"text": "Based"}

event: token
data: {"text": " on the uploaded document..."}

event: done
data: {"status": "completed", "workflow": ["retrieve", "generate", "stream"]}
```

Fallback event sequence:

```text
event: sources
data: {"sources": [...]}

event: token
data: {"text": "I could not find relevant information in the uploaded documents."}

event: done
data: {"status": "completed", "workflow": ["retrieve", "fallback"]}
```

Test streaming using curl:

```bash
curl -N -X POST "http://127.0.0.1:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"What information does the document provide?","top_k":3}'
```

## How Retrieval Works

The uploaded document is split into overlapping chunks. Each chunk is converted into a TF-IDF vector.

When a question is submitted:

1. The question is converted into a TF-IDF vector.
2. Cosine similarity is calculated against the indexed chunks.
3. The top-k chunks are selected.
4. The highest score is checked against the relevance threshold.
5. Relevant chunks are supplied to Claude as grounded context.
6. Claude is instructed to answer only from that context.
7. The response includes source filenames, chunk identifiers and similarity scores.

## Design Decisions

### Why use overlapping chunks?

Overlap helps preserve information that crosses chunk boundaries.

The current default uses 500-character chunks with a 100-character overlap. This improves context continuity, although it increases the number of indexed chunks.

### Why use TF-IDF for the MVP?

TF-IDF provides lightweight, fast and explainable lexical retrieval without requiring a large embedding-model download or an external vector database.

A future iteration can introduce dense semantic embeddings while retaining TF-IDF as part of a hybrid retrieval strategy.

### Why use a relevance threshold?

Top-k retrieval always returns the highest-ranking chunks, even when none of them are meaningfully related to the question.

The relevance threshold prevents weak matches from being sent to Claude and routes insufficient context to a deterministic fallback.

The current threshold is an initial heuristic. A production system should tune it using a representative evaluation dataset.

### Why return citations?

Filename and chunk identifiers make answers traceable. Users can inspect the retrieved passages and verify whether the response is grounded in the uploaded document.

### Why use top-k retrieval?

Sending only the most relevant chunks:

- Reduces unnecessary token usage
- Improves response latency
- Limits irrelevant context
- Reduces the chance of unsupported answers

A very small `top_k` may miss useful information, while a large value may introduce noise.

### Why use LangGraph?

The current graph is intentionally small, but it makes retrieval, conditional routing and generation explicit.

It also creates a clear extension path for:

- Query rewriting
- Metadata filtering
- Retrieval evaluation
- Reranking
- Human review
- Multi-step healthcare information workflows

### Why use SSE?

SSE provides unidirectional server-to-client streaming over HTTP and is well suited to incremental LLM output.

The API emits separate source, token, completion and error events, allowing clients to display partial responses while retaining structured metadata.

## Healthcare Safety

Claude is instructed to:

- Answer only from the supplied document context
- Avoid inventing medical facts
- Avoid providing a diagnosis
- State when the available context is insufficient
- Cite relevant document chunks

The API also returns a disclaimer explaining that responses are not a substitute for professional medical advice.

## Testing

Run:

```bash
pytest -q
```

The current test suite covers:

- Health endpoint
- TXT extraction
- Unsupported file rejection
- Overlapping chunk creation
- Invalid chunk configuration
- LangGraph generate routing
- LangGraph fallback routing
- Relevance-threshold decisions
- SSE JSON event formatting
- SSE source/token/done ordering
- SSE fallback behavior
- Claude streaming through a stubbed client

Current result:

```text
16 passed
```

External Claude calls are stubbed in automated tests to keep the suite deterministic and avoid consuming API credits.

## Current Limitations

- Documents and vectors are stored in memory and disappear after restart.
- Scanned PDFs require OCR and are not currently supported.
- TF-IDF is based on lexical rather than semantic similarity.
- Document ingestion is synchronous.
- The application runs as a single process.
- Authentication and tenant isolation are not implemented.
- The relevance threshold is heuristic rather than evaluation-driven.
- Retrieval-quality evaluation is not automated.
- The streaming route coordinates retrieval directly rather than executing the standard LangGraph generation node.

## Roadmap

- Dense semantic embeddings
- Persistent vector database
- Hybrid retrieval
- Metadata filtering
- Cross-encoder reranking
- Query rewriting
- Retrieval-quality evaluation
- Prompt-injection protection
- Authentication and tenant isolation
- Asynchronous document ingestion
- Streaming workflow events
- Docker deployment
- Cloud deployment
- Observability and tracing

## Author

**G Thiruvarangan Prasath**

Senior Full-Stack / Backend Engineer and Technical Lead with experience in Node.js, TypeScript, React, Python, healthcare platforms and AI-enabled applications.

- [LinkedIn](https://www.linkedin.com/in/thiru-prasath-7945aa63/)
- [GitHub](https://github.com/tprasath40)