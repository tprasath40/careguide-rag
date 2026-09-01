import os
from io import BytesIO
import json
from collections.abc import Iterator

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from fastapi.responses import StreamingResponse

load_dotenv()

app = FastAPI(
    title="CareGuide RAG API",
    description=(
        "A healthcare document question-answering application "
        "using TF-IDF retrieval, LangGraph and Claude"
    ),
    version="1.0.0",
)


stored_chunks: list[dict] = []
vectorizer: TfidfVectorizer | None = None
stored_vectors = None
MIN_RELEVANCE_SCORE = 0.05


class RagState(TypedDict):
    question: str
    top_k: int
    retrieved_chunks: list[dict]
    answer: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be greater than or equal to zero "
            "and smaller than chunk size"
        )

    cleaned_text = " ".join(text.split())
    chunks = []
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


def store_document(filename: str, chunks: list[str]) -> None:

    global stored_chunks
    global vectorizer
    global stored_vectors

    stored_chunks = [item for item in stored_chunks if item["filename"] != filename]

    for index, chunk in enumerate(chunks):
        stored_chunks.append({"filename": filename, "chunk_id": index, "text": chunk})

    all_texts = [item["text"] for item in stored_chunks]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    stored_vectors = vectorizer.fit_transform(all_texts)


def retrieve_chunks(question: str, top_k: int) -> list[dict]:

    if not stored_chunks or vectorizer is None or stored_vectors is None:
        raise HTTPException(
            status_code=400, detail="Upload a document before asking questions"
        )

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(question_vector, stored_vectors).flatten()

    result_count = min(top_k, len(stored_chunks))
    best_indices = scores.argsort()[::-1][:result_count]

    results = []

    for index in best_indices:
        chunk = stored_chunks[int(index)]

        results.append(
            {
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": round(float(scores[index]), 4),
            }
        )

    return results


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500, detail="ANTHROPIC_API_KEY is not configured"
        )

    context_parts = []

    for chunk in retrieved_chunks:
        source_label = f"[{chunk['filename']} - chunk {chunk['chunk_id']}]"

        context_parts.append(f"{source_label}\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"),
        max_tokens=500,
        system=(
            "You are CareGuide, a healthcare document assistant. "
            "Answer only using the supplied document context. "
            "Do not provide a diagnosis or invent medical facts. "
            "Cite relevant sources using their exact square-bracket "
            "labels. If the context does not contain the answer, say "
            "'I could not find that information in the uploaded documents.'"
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context}\n\n" f"QUESTION:\n{question}"
                ),
            }
        ],
    )

    return message.content[0].text


def format_sse_event(event: str, data: dict) -> str:

    return f"event: {event}\n" f"data: {json.dumps(data)}\n\n"


def retrieve_node(state: RagState) -> dict:
    chunks = retrieve_chunks(question=state["question"], top_k=state["top_k"])

    return {"retrieved_chunks": chunks}


def generate_node(state: RagState) -> dict:
    answer = generate_answer(
        question=state["question"], retrieved_chunks=state["retrieved_chunks"]
    )

    return {"answer": answer}


def fallback_node(state: RagState) -> dict:
    return {
        "answer": (
            "I could not find relevant information " "in the uploaded documents."
        )
    }


def route_after_retrieval(state: RagState) -> str:
    chunks = state["retrieved_chunks"]

    if not chunks:
        return "fallback"

    highest_score = chunks[0]["score"]

    if highest_score < MIN_RELEVANCE_SCORE:
        return "fallback"

    return "generate"


@app.get("/health")
def get_health():
    return {"status": "healthy", "service": "careguide-rag"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):

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
    store_document(filename, chunks)

    return {
        "filename": filename,
        "characters": len(text),
        "pages": pages,
        "chunks_created": len(chunks),
        "message": "Document indexed successfully",
    }


rag_workflow = StateGraph(RagState)

rag_workflow.add_node("retrieve", retrieve_node)
rag_workflow.add_node("generate", generate_node)
rag_workflow.add_node("fallback", fallback_node)

rag_workflow.add_edge(START, "retrieve")

rag_workflow.add_conditional_edges(
    "retrieve", route_after_retrieval, {"generate": "generate", "fallback": "fallback"}
)

rag_workflow.add_edge("generate", END)
rag_workflow.add_edge("fallback", END)

rag_graph = rag_workflow.compile()


@app.post("/query")
def query_document(request: QuestionRequest):

    result = rag_graph.invoke(
        {
            "question": request.question,
            "top_k": request.top_k,
            "retrieved_chunks": [],
            "answer": "",
        }
    )

    sources = [
        {
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "preview": chunk["text"][:200],
        }
        for chunk in result["retrieved_chunks"]
    ]

    used_fallback = not sources or sources[0]["score"] < MIN_RELEVANCE_SCORE

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": sources,
        "workflow": ["retrieve", "fallback" if used_fallback else "generate"],
        "disclaimer": (
            "This application provides document-based information only "
            "and is not a substitute for professional medical advice."
        ),
    }


@app.post("/query/stream")
def stream_query_document(request: QuestionRequest):

    return StreamingResponse(
        stream_query_events(question=request.question, top_k=request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def stream_claude_answer(question: str, retrieved_chunks: list[dict]) -> Iterator[str]:

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500, detail="ANTHROPIC_API_KEY is not configured"
        )

    context_parts = []

    for chunk in retrieved_chunks:
        source_label = f"[{chunk['filename']} - " f"chunk {chunk['chunk_id']}]"

        context_parts.append(f"{source_label}\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"),
        max_tokens=500,
        system=(
            "You are CareGuide, a healthcare document assistant. "
            "Answer only using the supplied document context. "
            "Do not provide a diagnosis or invent medical facts. "
            "Cite relevant sources using their exact square-bracket "
            "labels. If the context does not contain the answer, say "
            "'I could not find that information in the uploaded documents.'"
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context}\n\n" f"QUESTION:\n{question}"
                ),
            }
        ],
    ) as stream:

        for text in stream.text_stream:
            yield text


def stream_query_events(question: str, top_k: int) -> Iterator[str]:

    try:
        retrieved_chunks = retrieve_chunks(question=question, top_k=top_k)

        sources = [
            {
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "preview": chunk["text"][:200],
            }
            for chunk in retrieved_chunks
        ]

        yield format_sse_event(event="sources", data={"sources": sources})

        highest_score = sources[0]["score"] if sources else 0

        if highest_score < MIN_RELEVANCE_SCORE:
            fallback_answer = (
                "I could not find relevant information " "in the uploaded documents."
            )

            yield format_sse_event(event="token", data={"text": fallback_answer})

            yield format_sse_event(
                event="done",
                data={"status": "completed", "workflow": ["retrieve", "fallback"]},
            )

            return

        for token in stream_claude_answer(
            question=question, retrieved_chunks=retrieved_chunks
        ):
            yield format_sse_event(event="token", data={"text": token})

        yield format_sse_event(
            event="done",
            data={
                "status": "completed",
                "workflow": ["retrieve", "generate", "stream"],
            },
        )

    except HTTPException as error:
        yield format_sse_event(
            event="error", data={"status": error.status_code, "message": error.detail}
        )

    except anthropic.APIError:
        yield format_sse_event(
            event="error",
            data={
                "status": 502,
                "message": (
                    "The language model service " "could not complete the request"
                ),
            },
        )

    except Exception:
        yield format_sse_event(
            event="error", data={"status": 500, "message": "Unexpected streaming error"}
        )
