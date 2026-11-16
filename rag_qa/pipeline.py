"""The RAG pipeline: index documents, then answer questions from them.

Flow per question:

1. Embed the question with the same model used for the chunks.
2. Retrieve the most similar chunks from the FAISS store.
3. **Threshold check (in code, not in the prompt):** if the best chunk's
   cosine similarity is below ``similarity_threshold``, return
   ``"I don't know"`` without calling the LLM at all.
4. Otherwise pass the top chunks to the answer generator and return the
   answer together with citations for every chunk used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_documents
from .embeddings import Embedder, SentenceTransformerEmbedder
from .llm import AnswerGenerator, get_answer_generator
from .store import SearchResult, VectorStore

#: Default cosine-similarity threshold. Below this, the tool answers
#: "I don't know". Override in code, via the ``--threshold`` CLI flag, or
#: via the ``RAG_SIMILARITY_THRESHOLD`` environment variable.
DEFAULT_SIMILARITY_THRESHOLD = 0.35

DONT_KNOW = "I don't know"


@dataclass
class Answer:
    """The result of one question.

    Attributes:
        question: The question that was asked.
        text: The answer text, or ``"I don't know"`` when the tool
            declined to answer.
        citations: The retrieved chunks the answer is based on (empty
            when the tool declined).
        best_score: Highest similarity score among retrieved chunks.
        threshold: The threshold that was applied.
        answered: ``False`` when the threshold check declined to answer.
    """

    question: str
    text: str
    citations: list[SearchResult] = field(default_factory=list)
    best_score: float = 0.0
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    answered: bool = False


class RAGPipeline:
    """Index a folder of text documents and answer questions about them.

    Args:
        embedder: Embedding model wrapper (defaults to the local
            sentence-transformers model).
        generator: Answer generator (defaults to OpenAI when a key is
            configured, otherwise the offline extractive fallback).
        similarity_threshold: Minimum cosine similarity for the best
            chunk; below it the pipeline answers "I don't know".
        top_k: How many chunks to pass to the generator as context.
        chunk_size: Fixed chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        generator: AnswerGenerator | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.generator = generator or get_answer_generator()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = VectorStore()
        self.chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_documents(self, documents: dict[str, str]) -> int:
        """Chunk, embed, and store a ``{source_name: text}`` mapping.

        Returns:
            The number of chunks indexed.
        """
        self.chunks = chunk_documents(
            documents, self.chunk_size, self.chunk_overlap
        )
        embeddings = self.embedder.embed([c.text for c in self.chunks])
        self.store = VectorStore()
        self.store.add(self.chunks, embeddings)
        return len(self.chunks)

    def index_folder(self, folder: str | Path) -> int:
        """Index every ``.txt``/``.md`` file in ``folder``.

        Only plain sample/public documents should ever be indexed — never
        private files, per the task brief.
        """
        path = Path(folder)
        documents: dict[str, str] = {}
        for file in sorted(path.iterdir()):
            if file.suffix.lower() in (".txt", ".md") and file.is_file():
                documents[file.name] = file.read_text(encoding="utf-8")
        if not documents:
            raise ValueError(f"No .txt or .md documents found in {folder}.")
        return self.index_documents(documents)

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        """Answer ``question`` from the indexed documents.

        The similarity threshold is enforced here, in code: when the best
        retrieved chunk scores below it, the generator is never called
        and the answer is "I don't know".
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if len(self.store) == 0:
            raise RuntimeError("No documents indexed yet.")

        query_vector = self.embedder.embed([question])[0]
        results = self.store.search(query_vector, top_k=self.top_k)
        best_score = results[0].score if results else 0.0

        if not results or best_score < self.similarity_threshold:
            return Answer(
                question=question,
                text=DONT_KNOW,
                citations=[],
                best_score=best_score,
                threshold=self.similarity_threshold,
                answered=False,
            )

        context = [r.chunk.text for r in results]
        text = self.generator.answer(question, context)
        if text.strip().rstrip(".").lower() == "i don't know":
            return Answer(
                question=question,
                text=DONT_KNOW,
                citations=[],
                best_score=best_score,
                threshold=self.similarity_threshold,
                answered=False,
            )
        return Answer(
            question=question,
            text=text,
            citations=results,
            best_score=best_score,
            threshold=self.similarity_threshold,
            answered=True,
        )


def threshold_from_env(default: float = DEFAULT_SIMILARITY_THRESHOLD) -> float:
    """Read ``RAG_SIMILARITY_THRESHOLD`` from the environment, if set."""
    raw = os.environ.get("RAG_SIMILARITY_THRESHOLD", "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
