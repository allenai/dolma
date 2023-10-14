import argparse

from .executor import tokenize_in_parallel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--destination", required=True)
    ap.add_argument("--tokenizer-id", type=str, default="allenai/eleuther-ai-gpt-neox-20b-pii-special")
    ap.add_argument("--metadata-dir", type=str, default=None)
    ap.add_argument("--num-tokenizers", type=int, default=1)
    ap.add_argument("--num-writers", type=int, default=1)
    ap.add_argument("--max-size", type=int, default=1024 * 1024 * 1024)
    return ap.parse_args()


def main():
    args = parse_args()
    tokenize_in_parallel(
        sources=args.sources,
        destination=args.destination,
        tokenizer_name_or_path=args.tokenizer_id,
        metadata_dir=args.metadata_dir,
        num_tokenizers=args.num_tokenizers,
        num_writers=args.num_writers,
        max_size=args.max_size,
    )


if __name__ == "__main__":
    main()
