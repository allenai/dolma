"""Compare singleton and batched Dolma tokenizer throughput.

Usage:
    uv run python scripts/benchmark_tokenizer_batch.py tokenizer.json documents.txt
    uv run python scripts/benchmark_tokenizer_batch.py tokenizer.json documents.txt --backend gt
"""

import argparse
import time
from pathlib import Path

from dolma.tokenizer import Tokenizer, TokenizerBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path, help="HuggingFace-compatible tokenizer.json")
    parser.add_argument("documents", type=Path, help="UTF-8 file containing one document per line")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--backend",
        type=TokenizerBackend.parse,
        default=TokenizerBackend.huggingface,
        help="Fast tokenizer backend to benchmark: hf/huggingface (default) or gt/gigatoken.",
    )
    args = parser.parse_args()

    documents = [line.strip() for line in args.documents.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = Tokenizer.from_file(args.tokenizer, backend=args.backend)

    def run_singleton() -> int:
        return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in documents)

    def run_batched() -> int:
        return sum(
            len(ids)
            for start in range(0, len(documents), args.batch_size)
            for ids in tokenizer.encode_batch(documents[start : start + args.batch_size], add_special_tokens=False)
        )

    for label, run in (("singleton", run_singleton), ("batched", run_batched)):
        started = time.perf_counter()
        tokens = run()
        elapsed = time.perf_counter() - started
        print(f"{label:9} {tokens / elapsed:,.0f} tokens/s ({tokens:,} tokens in {elapsed:.3f}s)")


if __name__ == "__main__":
    main()
