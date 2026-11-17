"""Ask the assistant a question from the terminal (retrieval + LLM).

    python scripts/ask.py "Map her career using Where, How, Who and Why"
    python scripts/ask.py --mode direct "Where did she do her master's?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aboutme.assistant import answer  # noqa: E402
from aboutme.index import ensure_index  # noqa: E402
from aboutme.retrieval import Retriever  # noqa: E402
from aboutme.timeline import load_timeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["auto", "mindmap", "direct"], default="auto")
    parser.add_argument("-k", type=int, default=6)
    args = parser.parse_args()

    collection, _, embedder = ensure_index()
    result = answer(args.question, Retriever(collection, embedder), load_timeline(), k=args.k, mode=args.mode)
    if result.error:
        print(f"LLM unavailable: {result.error}\n")
    else:
        print(result.text, f"\n\n({result.model}, {result.mode})\n")
    print("Sources:")
    for i, hit in enumerate(result.hits, start=1):
        print(f"  [{i}] {hit.meta['doc_id']} · {hit.meta.get('section', '')}")


if __name__ == "__main__":
    main()
