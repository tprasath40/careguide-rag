"""Routing tests for the compiled LangGraph workflow.

Claude is never called: the generate node is monkeypatched, so these tests
exercise routing only.
"""

import pytest

from app.core.config import FALLBACK_ANSWER, MIN_RELEVANCE_SCORE
from app.services import claude_service
from app.services.retrieval_service import retrieval_service
from app.workflows import rag_workflow


def make_chunk(score: float) -> dict:
    return {
        "filename": "guide.txt",
        "chunk_id": 0,
        "text": "Healthcare document content",
        "score": score,
    }


def test_route_returns_fallback_without_chunks():
    state = {
        "question": "anything",
        "top_k": 3,
        "retrieved_chunks": [],
        "answer": "",
    }

    assert rag_workflow.route_after_retrieval(state) == "fallback"


def test_route_returns_fallback_below_relevance_threshold():
    state = {
        "question": "anything",
        "top_k": 3,
        "retrieved_chunks": [make_chunk(MIN_RELEVANCE_SCORE - 0.01)],
        "answer": "",
    }

    assert rag_workflow.route_after_retrieval(state) == "fallback"


def test_route_returns_generate_at_relevance_threshold():
    state = {
        "question": "anything",
        "top_k": 3,
        "retrieved_chunks": [make_chunk(MIN_RELEVANCE_SCORE)],
        "answer": "",
    }

    assert rag_workflow.route_after_retrieval(state) == "generate"


@pytest.fixture
def stub_generate(monkeypatch):
    calls: list[dict] = []

    def fake_generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
        calls.append({"question": question, "chunks": retrieved_chunks})
        return "Generated grounded answer"

    monkeypatch.setattr(claude_service, "generate_answer", fake_generate_answer)

    return calls


def stub_retrieval(monkeypatch, chunks: list[dict]) -> None:
    monkeypatch.setattr(
        retrieval_service,
        "retrieve_chunks",
        lambda question, top_k: chunks,
    )


def test_graph_routes_to_generate_for_relevant_chunks(monkeypatch, stub_generate):
    chunks = [make_chunk(0.8)]
    stub_retrieval(monkeypatch, chunks)

    result = rag_workflow.rag_graph.invoke(
        {
            "question": "What does the document say?",
            "top_k": 3,
            "retrieved_chunks": [],
            "answer": "",
        }
    )

    assert result["answer"] == "Generated grounded answer"
    assert result["retrieved_chunks"] == chunks
    assert len(stub_generate) == 1


def test_graph_routes_to_fallback_for_weak_chunks(monkeypatch, stub_generate):
    stub_retrieval(monkeypatch, [make_chunk(0.01)])

    result = rag_workflow.rag_graph.invoke(
        {
            "question": "What does the document say?",
            "top_k": 3,
            "retrieved_chunks": [],
            "answer": "",
        }
    )

    assert result["answer"] == FALLBACK_ANSWER
    assert stub_generate == []


def test_graph_routes_to_fallback_without_chunks(monkeypatch, stub_generate):
    stub_retrieval(monkeypatch, [])

    result = rag_workflow.rag_graph.invoke(
        {
            "question": "What does the document say?",
            "top_k": 3,
            "retrieved_chunks": [],
            "answer": "",
        }
    )

    assert result["answer"] == FALLBACK_ANSWER
    assert stub_generate == []
