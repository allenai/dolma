import argparse
import json
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

OLD_TOKENIZER_NAME = "allenai/gpt-neox-olmo-dolma-v1_5"
NEW_TOKENIZER_NAME = "allenai/gpt-neox-olmo-dolma-v1_5-digits"


def main(push_to_hub=False):
    old_tok = AutoTokenizer.from_pretrained(OLD_TOKENIZER_NAME)

    with TemporaryDirectory() as tmp_dir:
        old_tok.save_pretrained(tmp_dir)

        with open(f"{tmp_dir}/tokenizer.json", "r") as f:
            tokenizer_config = json.load(f)

            tokenizer_config["pre_tokenizer"] = {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Digits", "individual_digits": True},
                    tokenizer_config["pre_tokenizer"],
                ],
            }

        with open(f"{tmp_dir}/tokenizer.json", "w") as f:
            json.dump(tokenizer_config, f)

        new_tok = AutoTokenizer.from_pretrained(tmp_dir)

    hello_world = "Hello world<|endoftext|>"
    new_enc = new_tok.encode(hello_world)
    old_enc = old_tok.encode(hello_world)
    assert len(new_enc) == len(old_enc)
    assert new_enc == old_enc

    hello_digits = "Hello *1234* world<|endoftext|>"
    new_enc = new_tok.encode(hello_digits)
    old_enc = old_tok.encode(hello_digits)
    assert len(new_enc) == len(old_enc) + 3
    assert new_enc[:2] == old_enc[:2]
    assert new_enc[2:6] == [old_tok.vocab[d] for d in "1234"]
    assert old_enc[2:3] == [old_tok.vocab["1234"]]
    assert new_enc[6:] == old_enc[3:]

    if push_to_hub:
        print("Pushing to hub...")
        new_tok.push_to_hub(NEW_TOKENIZER_NAME)
        print(f"tokenizer available at: https://huggingface.co/{NEW_TOKENIZER_NAME}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()
    main(args.push_to_hub)
