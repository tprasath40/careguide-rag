from typing import TypedDict

from pydantic import BaseModel, Field


class RagState(TypedDict):
    question: str
    top_k: int
    retrieved_chunks: list[dict]
    answer: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)
