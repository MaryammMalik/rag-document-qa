"""Fixed-size text chunking.

Each document is split into chunks of at most ``chunk_size`` characters,
with a small overlap between consecutive chunks so a sentence that falls
on a boundary still appears whole in at least one chunk. Splits prefer
word boundaries — a chunk never cuts a word in half when it can avoid it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """One chunk of a source document.

    Attributes:
        chunk_id: Stable identifier, e.g. ``"python_basics#2"``.
        source: File name of the document the chunk came from.
        index: Zero-based position of the chunk inside its document.
        text: The chunk's text.
    """

    chunk_id: str
    source: str
    index: int
    text: str


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split ``text`` into fixed-size chunks with overlap.

    Args:
        text: The document text to split.
        source: Name of the document (used to build chunk IDs).
        chunk_size: Maximum characters per chunk.
        overlap: Characters repeated between consecutive chunks.

    Returns:
        A list of :class:`Chunk` objects in document order.

    Raises:
        ValueError: If ``chunk_size`` is not positive or ``overlap`` is
            not smaller than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    cleaned = " ".join(text.split())
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        # Prefer to end on a word boundary.
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}#{index}",
                    source=source,
                    index=index,
                    text=piece,
                )
            )
            index += 1
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
        # Start the next chunk on a word boundary too.
        boundary = cleaned.find(" ", start)
        if 0 <= boundary < end:
            start = boundary + 1
    return chunks


def chunk_documents(
    documents: dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk every document in a ``{source_name: text}`` mapping."""
    all_chunks: list[Chunk] = []
    for source, text in documents.items():
        all_chunks.extend(chunk_text(text, source, chunk_size, overlap))
    return all_chunks
