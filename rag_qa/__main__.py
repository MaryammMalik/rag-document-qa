"""Allow running the tool with ``python -m rag_qa``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
