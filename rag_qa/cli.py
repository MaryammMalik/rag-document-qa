"""Command-line interface for the RAG Document Q&A tool.

Usage:
    python -m rag_qa "When is the one-month internship deadline?"
    python -m rag_qa --interactive
    python -m rag_qa "What is chunking?" --threshold 0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import DEFAULT_SIMILARITY_THRESHOLD, Answer, RAGPipeline

DEFAULT_DOCS = Path(__file__).resolve().parent.parent / "data"


def format_answer(answer: Answer, generator_name: str) -> str:
    """Render an :class:`Answer` with its citations for the terminal."""
    lines = [f"Answer: {answer.text}"]
    if answer.answered and answer.citations:
        lines.append("\nSources used:")
        for result in answer.citations:
            chunk = result.chunk
            lines.append(
                f"  - {chunk.chunk_id} (similarity {result.score:.2f}): "
                f"{chunk.text[:110]}..."
            )
    elif not answer.answered:
        lines.append(
            f"(Best matching chunk scored {answer.best_score:.2f}, below "
            f"the threshold of {answer.threshold:.2f} — the documents do "
            "not seem to contain this answer.)"
        )
    lines.append(f"[answered by: {generator_name}]")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="rag-qa",
        description="Ask questions about a folder of text documents (RAG).",
    )
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument(
        "--docs",
        default=str(DEFAULT_DOCS),
        help="Folder of .txt/.md documents to index. Default: ./data",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Similarity threshold below which the tool answers "
            f"\"I don't know\". Default: {DEFAULT_SIMILARITY_THRESHOLD} "
            "(or RAG_SIMILARITY_THRESHOLD from the environment)."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive question loop.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    from .pipeline import threshold_from_env

    threshold = (
        args.threshold
        if args.threshold is not None
        else threshold_from_env()
    )
    pipeline = RAGPipeline(similarity_threshold=threshold)

    try:
        count = pipeline.index_folder(args.docs)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Indexed {count} chunks from {args.docs} "
        f"(threshold {threshold:.2f}, generator: {pipeline.generator.name})"
    )

    if args.interactive:
        print("Ask a question, or type 'exit' to quit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in ("exit", "quit"):
                break
            if question:
                answer = pipeline.ask(question)
                print(format_answer(answer, pipeline.generator.name))
        return 0

    question = " ".join(args.question).strip()
    if not question:
        print(
            "Error: provide a question or use --interactive.",
            file=sys.stderr,
        )
        return 2
    answer = pipeline.ask(question)
    print(format_answer(answer, pipeline.generator.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
