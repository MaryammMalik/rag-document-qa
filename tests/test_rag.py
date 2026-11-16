"""Tests for the RAG pipeline.

The suite uses a tiny deterministic bag-of-words embedder and a fake
answer generator, so it runs offline in milliseconds. The behaviours
under test are the ones the brief calls out: fixed-size chunking,
similarity search, the **configurable threshold** that produces
"I don't know", and chunk citations.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_qa.chunking import chunk_documents, chunk_text
from rag_qa.pipeline import DONT_KNOW, RAGPipeline
from rag_qa.store import VectorStore

VOCAB = [
    "deadline", "internship", "november", "certificate", "python",
    "function", "chunk", "embedding", "weather", "rain",
]


class BagOfWordsEmbedder:
    """Deterministic test embedder over a fixed vocabulary."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [lowered.count(word) for word in VOCAB], dtype=np.float32
            )
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.array(vectors, dtype=np.float32)


class FakeGenerator:
    """Records calls and answers from the first context chunk."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, context_chunks: list[str]) -> str:
        self.calls += 1
        return f"Generated from: {context_chunks[0][:40]}"


DOCS = {
    "internship.txt": (
        "The internship deadline is in November. A certificate is "
        "issued after the internship ends."
    ),
    "python.txt": (
        "A Python function is defined with the def keyword. Every "
        "function can return a value."
    ),
}


def make_pipeline(threshold: float = 0.3) -> tuple[RAGPipeline, FakeGenerator]:
    generator = FakeGenerator()
    pipeline = RAGPipeline(
        embedder=BagOfWordsEmbedder(),
        generator=generator,
        similarity_threshold=threshold,
        chunk_size=200,
        chunk_overlap=20,
    )
    pipeline.index_documents(DOCS)
    return pipeline, generator


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def test_chunking_splits_and_labels() -> None:
    text = "word " * 300
    chunks = chunk_text(text.strip(), "doc.txt", chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.chunk_id for c in chunks] == [
        f"doc.txt#{i}" for i in range(len(chunks))
    ]
    assert all(c.text for c in chunks)


def test_chunking_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", "d.txt", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", "d.txt", chunk_size=100, overlap=100)


def test_chunk_documents_covers_all_sources() -> None:
    chunks = chunk_documents(DOCS, chunk_size=200, overlap=20)
    assert {c.source for c in chunks} == {"internship.txt", "python.txt"}


# ----------------------------------------------------------------------
# Vector store
# ----------------------------------------------------------------------
def test_store_search_orders_by_similarity() -> None:
    embedder = BagOfWordsEmbedder()
    chunks = chunk_documents(DOCS, chunk_size=200, overlap=20)
    store = VectorStore()
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    results = store.search(embedder.embed(["deadline november"])[0], top_k=1)
    assert results[0].chunk.source == "internship.txt"
    assert results[0].score > 0.5


def test_store_empty_search() -> None:
    store = VectorStore()
    assert store.search(np.zeros(4, dtype=np.float32)) == []


# ----------------------------------------------------------------------
# Pipeline behaviour — the heart of Task 7
# ----------------------------------------------------------------------
def test_answerable_question_is_answered_with_citations() -> None:
    pipeline, generator = make_pipeline()
    answer = pipeline.ask("When is the internship deadline in November?")
    assert answer.answered is True
    assert answer.text.startswith("Generated from:")
    assert generator.calls == 1
    assert answer.citations
    assert answer.citations[0].chunk.source == "internship.txt"


def test_unanswerable_question_declines_without_calling_generator() -> None:
    pipeline, generator = make_pipeline()
    answer = pipeline.ask("What will the weather rain be tomorrow?")
    # "weather"/"rain" appear in the vocabulary but in NO document chunk
    # above, so retrieval still runs — the threshold must stop the answer.
    assert answer.answered is False
    assert answer.text == DONT_KNOW
    assert answer.citations == []
    assert generator.calls == 0


def test_threshold_is_configurable() -> None:
    strict, generator = make_pipeline(threshold=0.999)
    answer = strict.ask("When is the internship deadline in November?")
    assert answer.answered is False
    assert answer.text == DONT_KNOW
    assert generator.calls == 0


def test_generator_saying_dont_know_is_respected() -> None:
    class ShyGenerator(FakeGenerator):
        def answer(self, question: str, context_chunks: list[str]) -> str:
            self.calls += 1
            return "I don't know"

    pipeline = RAGPipeline(
        embedder=BagOfWordsEmbedder(),
        generator=ShyGenerator(),
        similarity_threshold=0.3,
        chunk_size=200,
        chunk_overlap=20,
    )
    pipeline.index_documents(DOCS)
    answer = pipeline.ask("When is the internship deadline in November?")
    assert answer.answered is False
    assert answer.text == DONT_KNOW


def test_empty_question_rejected() -> None:
    pipeline, _ = make_pipeline()
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_ask_before_indexing_rejected() -> None:
    pipeline = RAGPipeline(
        embedder=BagOfWordsEmbedder(), generator=FakeGenerator()
    )
    with pytest.raises(RuntimeError):
        pipeline.ask("anything?")
