"""Text embeddings.

The default embedder uses the **free, local** ``sentence-transformers``
library (model ``all-MiniLM-L6-v2``), exactly as the task brief suggests —
no API key and no document ever leaves the machine. Embeddings are
L2-normalised, so a plain inner product equals cosine similarity.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that can turn texts into normalised embedding vectors."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float array of normalised embeddings."""
        ...


class SentenceTransformerEmbedder:
    """Embedder backed by a local sentence-transformers model.

    Args:
        model_name: Hugging Face model name. The model is downloaded once
            on first use and cached locally afterwards.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` and L2-normalise the vectors."""
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)
