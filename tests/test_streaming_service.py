"""SSE formatting and stream-event ordering tests.

Claude streaming is monkeypatched, so no real API call is made.
"""

import json

from fastapi import HTTPException

from app.core.config import FALLBACK_ANSWER
from app.services import claude_service, streaming_service
from app.services.retrieval_service import retrieval_service


def make_chunk(score: float) -> dict:
    return {
        "filename": "guide.txt",
        "chunk_id": 0,
        "text": "Healthcare document content",
        "score": score,
    }


def parse_events(raw_events: list[str]) -> list[tuple[str, dict]]:
    parsed: list[tuple[str, dict]] = []

    for raw in raw_events:
        assert raw.endswith("\n\n")

        event_line, data_line = raw.rstrip("\n").split("\n")

        parsed.append(
            (
                event_line.removeprefix("event: "),
                json.loads(data_line.removeprefix("data: ")),
            )
        )

    return parsed


def test_format_sse_event_shape():
    formatted = streaming_service.format_sse_event(
        event="token", data={"text": "hello"}
    )

    assert formatted == 'event: token\ndata: {"text": "hello"}\n\n'


def test_format_sse_event_serialises_nested_data():
    formatted = streaming_service.format_sse_event(
        event="sources", data={"sources": [{"chunk_id": 0}]}
    )

    assert formatted == 'event: sources\ndata: {"sources": [{"chunk_id": 0}]}\n\n'


def test_stream_emits_sources_tokens_then_done(monkeypatch):
    monkeypatch.setattr(
        retrieval_service,
        "retrieve_chunks",
        lambda question, top_k: [make_chunk(0.8)],
    )

    monkeypatch.setattr(
        claude_service,
        "stream_claude_answer",
        lambda question, retrieved_chunks: iter(["Take ", "one ", "tablet."]),
    )

    events = parse_events(
        list(streaming_service.stream_query_events(question="dosage?", top_k=3))
    )

    assert [name for name, _ in events] == [
        "sources",
        "token",
        "token",
        "token",
        "done",
    ]

    assert events[0][1]["sources"][0]["filename"] == "guide.txt"
    assert "".join(data["text"] for _, data in events[1:4]) == "Take one tablet."
    assert events[-1][1] == {
        "status": "completed",
        "workflow": ["retrieve", "generate", "stream"],
    }


def test_stream_uses_fallback_below_relevance_threshold(monkeypatch):
    monkeypatch.setattr(
        retrieval_service,
        "retrieve_chunks",
        lambda question, top_k: [make_chunk(0.01)],
    )

    def fail_if_called(question, retrieved_chunks):
        raise AssertionError("Claude must not be called on the fallback path")

    monkeypatch.setattr(claude_service, "stream_claude_answer", fail_if_called)

    events = parse_events(
        list(streaming_service.stream_query_events(question="dosage?", top_k=3))
    )

    assert [name for name, _ in events] == ["sources", "token", "done"]
    assert events[1][1] == {"text": FALLBACK_ANSWER}
    assert events[-1][1] == {
        "status": "completed",
        "workflow": ["retrieve", "fallback"],
    }


def test_stream_emits_error_event_for_http_exception(monkeypatch):
    def raise_http_exception(question, top_k):
        raise HTTPException(
            status_code=400, detail="Upload a document before asking questions"
        )

    monkeypatch.setattr(retrieval_service, "retrieve_chunks", raise_http_exception)

    events = parse_events(
        list(streaming_service.stream_query_events(question="dosage?", top_k=3))
    )

    assert events == [
        (
            "error",
            {"status": 400, "message": "Upload a document before asking questions"},
        )
    ]
