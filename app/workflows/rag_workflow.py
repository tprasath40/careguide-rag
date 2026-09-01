from langgraph.graph import END, START, StateGraph

from app.core.config import FALLBACK_ANSWER, MIN_RELEVANCE_SCORE
from app.models.schemas import RagState
from app.services import claude_service
from app.services.retrieval_service import retrieval_service


def retrieve_node(state: RagState) -> dict:
    chunks = retrieval_service.retrieve_chunks(
        question=state["question"], top_k=state["top_k"]
    )

    return {"retrieved_chunks": chunks}


def generate_node(state: RagState) -> dict:
    answer = claude_service.generate_answer(
        question=state["question"], retrieved_chunks=state["retrieved_chunks"]
    )

    return {"answer": answer}


def fallback_node(state: RagState) -> dict:
    return {"answer": FALLBACK_ANSWER}


def route_after_retrieval(state: RagState) -> str:
    chunks = state["retrieved_chunks"]

    if not chunks:
        return "fallback"

    highest_score = chunks[0]["score"]

    if highest_score < MIN_RELEVANCE_SCORE:
        return "fallback"

    return "generate"


def create_rag_graph():
    """Build and compile the retrieve -> generate/fallback -> END workflow."""

    rag_workflow = StateGraph(RagState)

    rag_workflow.add_node("retrieve", retrieve_node)
    rag_workflow.add_node("generate", generate_node)
    rag_workflow.add_node("fallback", fallback_node)

    rag_workflow.add_edge(START, "retrieve")

    rag_workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"generate": "generate", "fallback": "fallback"},
    )

    rag_workflow.add_edge("generate", END)
    rag_workflow.add_edge("fallback", END)

    return rag_workflow.compile()


rag_graph = create_rag_graph()
