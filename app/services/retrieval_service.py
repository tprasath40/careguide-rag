from typing import Any

from fastapi import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalService:
    """In-memory TF-IDF index with cosine-similarity top-k retrieval."""

    def __init__(self) -> None:
        self.stored_chunks: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.stored_vectors: Any = None

    def store_document(self, filename: str, chunks: list[str]) -> None:

        self.stored_chunks = [
            item for item in self.stored_chunks if item["filename"] != filename
        ]

        for index, chunk in enumerate(chunks):
            self.stored_chunks.append(
                {"filename": filename, "chunk_id": index, "text": chunk}
            )

        all_texts = [item["text"] for item in self.stored_chunks]

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

        self.stored_vectors = self.vectorizer.fit_transform(all_texts)

    def retrieve_chunks(self, question: str, top_k: int) -> list[dict]:

        if (
            not self.stored_chunks
            or self.vectorizer is None
            or self.stored_vectors is None
        ):
            raise HTTPException(
                status_code=400, detail="Upload a document before asking questions"
            )

        question_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(question_vector, self.stored_vectors).flatten()

        result_count = min(top_k, len(self.stored_chunks))
        best_indices = scores.argsort()[::-1][:result_count]

        results: list[dict] = []

        for index in best_indices:
            chunk = self.stored_chunks[int(index)]

            results.append(
                {
                    "filename": chunk["filename"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": round(float(scores[index]), 4),
                }
            )

        return results


def build_sources(retrieved_chunks: list[dict]) -> list[dict]:
    """Shape retrieved chunks into the citation payload returned by the API."""

    return [
        {
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "preview": chunk["text"][:200],
        }
        for chunk in retrieved_chunks
    ]


# The single in-memory retrieval index shared by uploads, queries and streaming.
retrieval_service = RetrievalService()
