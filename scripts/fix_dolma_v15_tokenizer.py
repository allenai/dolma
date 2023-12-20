import argparse
import json
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

OG_TOKENIZER_NAME = "eleutherai/gpt-neox-20b"
OLD_TOKENIZER_NAME = "allenai/eleuther-ai-gpt-neox-20b-pii-special"
NEW_TOKENIZER_NAME = "allenai/gpt-neox-olmo-dolma-v1_5"
EMAIL_SPECIAL_TOKEN = "|||EMAIL_ADDRESS|||"
EMAIL_SPECIAL_TOKEN_ID = 50277
PHONE_SPECIAL_TOKEN = "|||PHONE_NUMBER|||"
PHONE_SPECIAL_TOKEN_ID = 50278
IP_SPECIAL_TOKEN = "|||IP_ADDRESS|||"
IP_SPECIAL_TOKEN_ID = 50279
EOS_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 0


def main(push_to_hub=False):
    og_tok = AutoTokenizer.from_pretrained(OG_TOKENIZER_NAME)
    old_tok = AutoTokenizer.from_pretrained(OLD_TOKENIZER_NAME)

    assert old_tok.eos_token == EOS_TOKEN
    assert old_tok.bos_token == EOS_TOKEN
    assert old_tok.unk_token == EOS_TOKEN

    vocab = old_tok.get_vocab()

    assert len(vocab) == 50280
    assert vocab[old_tok.eos_token] == old_tok.eos_token_id == EOS_TOKEN_ID
    assert vocab[old_tok.bos_token] == old_tok.bos_token_id == EOS_TOKEN_ID
    assert vocab[old_tok.unk_token] == old_tok.unk_token_id == EOS_TOKEN_ID
    assert vocab[IP_SPECIAL_TOKEN] == IP_SPECIAL_TOKEN_ID
    assert vocab[EMAIL_SPECIAL_TOKEN] == 50277
    assert vocab[PHONE_SPECIAL_TOKEN] == 50278

    with TemporaryDirectory() as tmp_dir:
        old_tok.save_pretrained(tmp_dir)

        with open(f"{tmp_dir}/tokenizer.json", "r") as f:
            tokenizer_config = json.load(f)

        for token_config in tokenizer_config["added_tokens"]:
            if token_config["content"] == EOS_TOKEN:
                token_config["id"] = IP_SPECIAL_TOKEN_ID
            elif token_config["content"] == IP_SPECIAL_TOKEN:
                token_config["id"] = EOS_TOKEN_ID
            tokenizer_config["model"]["vocab"][token_config["content"]] = token_config["id"]
        tokenizer_config["added_tokens"] = sorted(tokenizer_config["added_tokens"], key=lambda x: x["id"])
        tokenizer_config["model"]["vocab"] = {
            k: v for k, v in sorted(tokenizer_config["model"]["vocab"].items(), key=lambda x: x[1])
        }

        with open(f"{tmp_dir}/tokenizer.json", "w") as f:
            json.dump(tokenizer_config, f)

        new_tok = AutoTokenizer.from_pretrained(tmp_dir)

    # check if the swap worked
    new_vocab = new_tok.get_vocab()
    assert new_vocab[new_tok.eos_token] == new_tok.eos_token_id == IP_SPECIAL_TOKEN_ID
    assert new_vocab[new_tok.bos_token] == new_tok.bos_token_id == IP_SPECIAL_TOKEN_ID
    assert new_vocab[new_tok.unk_token] == new_tok.unk_token_id == IP_SPECIAL_TOKEN_ID
    assert new_vocab[IP_SPECIAL_TOKEN] == EOS_TOKEN_ID

    assert new_tok.encode("|||IP_ADDRESS|||") == [EOS_TOKEN_ID]
    assert new_tok.encode("|||EMAIL_ADDRESS|||") == [EMAIL_SPECIAL_TOKEN_ID]
    assert new_tok.encode("|||PHONE_NUMBER|||") == [PHONE_SPECIAL_TOKEN_ID]

    hello_world = "Hello world<|endoftext|>"
    new_enc = new_tok.encode(hello_world)
    old_enc = old_tok.encode(hello_world)
    og_enc = og_tok.encode(hello_world)
    assert len(new_enc) == len(old_enc)
    assert new_enc[:-1] == old_enc[:-1]
    assert new_enc[:-1] == og_enc[:-1]
    assert new_enc[-1] == IP_SPECIAL_TOKEN_ID
    assert old_enc[-1] == EOS_TOKEN_ID
    assert og_enc[-1] == EOS_TOKEN_ID

    text_with_split = "Heeeeellllooooo subwords!!!<|endoftext|>"
    new_enc = new_tok.encode(text_with_split)
    old_enc = old_tok.encode(text_with_split)
    og_enc = og_tok.encode(text_with_split)
    assert len(new_enc) == len(old_enc)
    assert new_enc[:-1] == old_enc[:-1]
    assert new_enc[:-1] == og_enc[:-1]
    assert new_enc[-1] == IP_SPECIAL_TOKEN_ID
    assert old_enc[-1] == EOS_TOKEN_ID
    assert og_enc[-1] == EOS_TOKEN_ID

    for token_id in range(0, len(og_tok.get_vocab())):
        og_vocab_entry = og_tok.convert_ids_to_tokens(token_id)
        new_vocab_entry = new_tok.convert_ids_to_tokens(token_id)
        if token_id == EOS_TOKEN_ID:
            assert new_vocab_entry == IP_SPECIAL_TOKEN
            assert og_vocab_entry == EOS_TOKEN
        else:
            err = f"{token_id}: `{og_vocab_entry}` != `{new_vocab_entry}`"
            assert og_vocab_entry == new_vocab_entry, err

    for token_id in range(0, len(old_tok.get_vocab())):
        old_vocab_entry = old_tok.convert_ids_to_tokens(token_id)
        new_vocab_entry = new_tok.convert_ids_to_tokens(token_id)
        if token_id == EOS_TOKEN_ID:
            assert new_vocab_entry == IP_SPECIAL_TOKEN
            assert old_vocab_entry == EOS_TOKEN
        elif token_id == IP_SPECIAL_TOKEN_ID:
            assert new_vocab_entry == EOS_TOKEN
            assert old_vocab_entry == IP_SPECIAL_TOKEN
        else:
            err = f"{token_id}: `{old_vocab_entry}` != `{new_vocab_entry}`"
            assert old_vocab_entry == new_vocab_entry, err

    masked_text = "<|padding|>Hello my phone number is |||PHONE_NUMBER||| bye <|endoftext|>"
    new_enc = new_tok.encode(masked_text)
    old_enc = old_tok.encode(masked_text)
    assert len(new_enc) == len(old_enc)
    assert new_enc[:-1] == old_enc[:-1]
    assert new_enc[-1] == IP_SPECIAL_TOKEN_ID
    assert old_enc[-1] == EOS_TOKEN_ID

    if push_to_hub:
        print("Pushing to hub...")
        new_tok.push_to_hub(NEW_TOKENIZER_NAME)
        print(f"tokenizer available at: https://huggingface.co/{NEW_TOKENIZER_NAME}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()
    main(args.push_to_hub)
