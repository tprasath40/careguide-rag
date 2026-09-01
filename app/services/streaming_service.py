import json
from collections.abc import Iterator

import anthropic
from fastapi import HTTPException

from app.core.config import FALLBACK_ANSWER, MIN_RELEVANCE_SCORE
from app.services import claude_service
from app.services.retrieval_service import build_sources, retrieval_service


def format_sse_event(event: str, data: dict) -> str:

    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_query_events(question: str, top_k: int) -> Iterator[str]:

    try:
        retrieved_chunks = retrieval_service.retrieve_chunks(
            question=question, top_k=top_k
        )

        sources = build_sources(retrieved_chunks)

        yield format_sse_event(event="sources", data={"sources": sources})

        highest_score = sources[0]["score"] if sources else 0

        if highest_score < MIN_RELEVANCE_SCORE:
            yield format_sse_event(event="token", data={"text": FALLBACK_ANSWER})

            yield format_sse_event(
                event="done",
                data={"status": "completed", "workflow": ["retrieve", "fallback"]},
            )

            return

        for token in claude_service.stream_claude_answer(
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
                "message": "The language model service could not complete the request",
            },
        )

    except Exception:
        yield format_sse_event(
            event="error", data={"status": 500, "message": "Unexpected streaming error"}
        )
