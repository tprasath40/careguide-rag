from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import MEDICAL_DISCLAIMER, MIN_RELEVANCE_SCORE
from app.models.schemas import QuestionRequest
from app.services.retrieval_service import build_sources
from app.services.streaming_service import stream_query_events
from app.workflows.rag_workflow import rag_graph

router = APIRouter()


@router.post("/query")
def query_document(request: QuestionRequest) -> dict:

    result = rag_graph.invoke(
        {
            "question": request.question,
            "top_k": request.top_k,
            "retrieved_chunks": [],
            "answer": "",
        }
    )

    sources = build_sources(result["retrieved_chunks"])

    used_fallback = not sources or sources[0]["score"] < MIN_RELEVANCE_SCORE

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": sources,
        "workflow": ["retrieve", "fallback" if used_fallback else "generate"],
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@router.post("/query/stream")
def stream_query_document(request: QuestionRequest) -> StreamingResponse:

    return StreamingResponse(
        stream_query_events(question=request.question, top_k=request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
