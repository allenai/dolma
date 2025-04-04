import json
import shlex
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

MOD_DOLMA_TOKENIZER = "allenai/dolma2-tokenizer-U10F0F0"
OG_DOLMA_TOKENIZER = "allenai/dolma2-tokenizer"

CMD = """
cargo run -- \
    --inputs 'data/input/*.jsonl' \
    --output data/output \
    --substitutions '<|endoftext|>=<|􏃰endoftext|>' \
    --substitutions '<|fim_prefix|>=<|􏃰fim_prefix|>' \
    --substitutions '<|fim_middle|>=<|􏃰fim_middle|>' \
    --substitutions '<|fim_suffix|>=<|􏃰fim_suffix|>' \
    --substitutions '<|im_start|>=<|􏃰im_start|>' \
    --substitutions '<|im_end|>=<|􏃰im_end|>' \
    --substitutions '<|endofprompt|>=<|􏃰endofprompt|>' \
    --substitutions '<|pad|>=<|􏃰pad|>'
"""


def _find_rust_root() -> Path:
    rust_root = Path(__file__)
    while True:
        if rust_root == Path("/"):
            raise FileNotFoundError("Could not find rust root")
        if (rust_root / "Cargo.toml").exists():
            return rust_root
        rust_root = rust_root.parent


def test_sanitizer():
    og_tok = AutoTokenizer.from_pretrained(OG_DOLMA_TOKENIZER)
    mod_tok = AutoTokenizer.from_pretrained(MOD_DOLMA_TOKENIZER)

    root_dir = _find_rust_root()
    subprocess.run(shlex.split(CMD), check=True, cwd=root_dir)

    input_dir = root_dir / "data" / "input"
    output_dir = root_dir / "data" / "output"

    for input_file in input_dir.glob("*.jsonl"):
        with input_file.open("r") as f:
            input_docs = [json.loads(line) for line in f]

        output_file = output_dir / input_file.name
        with output_file.open("r") as f:
            output_docs = [json.loads(line) for line in f]

        for input_doc, output_doc in zip(input_docs, output_docs):
            input_tokens = og_tok.tokenize(input_doc["text"], split_special_tokens=True)
            output_tokens = mod_tok.tokenize(output_doc["text"], split_special_tokens=False)

            assert input_tokens == output_tokens


if __name__ == "__main__":
    test_sanitizer()
