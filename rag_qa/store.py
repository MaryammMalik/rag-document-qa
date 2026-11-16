"""FAISS vector store for chunk embeddings.

Embeddings are L2-normalised (see :mod:`rag_qa.embeddings`), so FAISS's
inner-product index returns true cosine-similarity scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from .chunking import Chunk


@dataclass
class SearchResult:
    """One retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """A minimal FAISS index over document chunks."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embedding vectors to the index.

        Raises:
            ValueError: If the number of vectors does not match the
                number of chunks.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks and embeddings count mismatch.")
        if self._index is None:
            self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks to a query vector.

        ``query_embedding`` must be a normalised 1-D array. Results are
        ordered by descending cosine similarity.
        """
        if self._index is None or not self._chunks:
            return []
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(vector, min(top_k, len(self._chunks)))
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(
                    SearchResult(chunk=self._chunks[int(idx)], score=float(score))
                )
        return results
