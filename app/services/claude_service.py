from collections.abc import Iterator

import anthropic
from fastapi import HTTPException

from app.core.config import (
    CLAUDE_MAX_TOKENS,
    HEALTHCARE_SYSTEM_PROMPT,
    get_anthropic_api_key,
    get_claude_model,
)


def _require_api_key() -> str:

    api_key = get_anthropic_api_key()

    if not api_key:
        raise HTTPException(
            status_code=500, detail="ANTHROPIC_API_KEY is not configured"
        )

    return api_key


def build_context(retrieved_chunks: list[dict]) -> str:
    """Join retrieved chunks into a citation-labelled context block."""

    context_parts: list[str] = []

    for chunk in retrieved_chunks:
        source_label = f"[{chunk['filename']} - chunk {chunk['chunk_id']}]"

        context_parts.append(f"{source_label}\n{chunk['text']}")

    return "\n\n".join(context_parts)


def build_user_message(question: str, context: str) -> str:

    return f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:

    api_key = _require_api_key()

    context = build_context(retrieved_chunks)

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=get_claude_model(),
        max_tokens=CLAUDE_MAX_TOKENS,
        system=HEALTHCARE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_message(question, context),
            }
        ],
    )

    return message.content[0].text


def stream_claude_answer(question: str, retrieved_chunks: list[dict]) -> Iterator[str]:

    api_key = _require_api_key()

    context = build_context(retrieved_chunks)

    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model=get_claude_model(),
        max_tokens=CLAUDE_MAX_TOKENS,
        system=HEALTHCARE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_message(question, context),
            }
        ],
    ) as stream:

        for text in stream.text_stream:
            yield text
