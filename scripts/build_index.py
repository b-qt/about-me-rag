"""Build (or refresh) the vector index from knowledge/ and optionally test a query.

    python scripts/build_index.py            # rebuild only if knowledge/ changed
    python scripts/build_index.py --force    # always rebuild
    python scripts/build_index.py --query "Where did she study?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aboutme.embeddings import get_embedder  # noqa: E402
from aboutme.index import build_index, ensure_index  # noqa: E402
from aboutme.retrieval import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    parser.add_argument("--query", help="print the passages retrieved for this question")
    parser.add_argument("-k", type=int, default=6)
    args = parser.parse_args()

    embedder = get_embedder()
    if args.force:
        build_index(embedder)
    collection, stats, embedder = ensure_index(embedder)
    print(f"{'Rebuilt' if stats.rebuilt or args.force else 'Up to date'}: {stats.documents} documents, "
          f"{stats.chunks} chunks {stats.by_type} ({stats.embedder})")

    if args.query:
        for i, hit in enumerate(Retriever(collection, embedder).search(args.query, k=args.k), start=1):
            print(f"\n[{i}] {hit.meta['doc_id']} · {hit.meta.get('section', '')} (score {hit.score:.4f})")
            print("    " + hit.text.replace("\n", "\n    ")[:500])


if __name__ == "__main__":
    main()
