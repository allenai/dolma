import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

import numpy
import smart_open
from tokenizers import Tokenizer as BaseTokenizer

from dolma.cli.__main__ import main
from dolma.tokenizer import Tokenizer

TEST_DIR = Path(__file__).parent.parent.resolve()


LLAMA_TOKENIZER = {
    "filename": f"{TEST_DIR}/data/tokenizer/llama-test-tokenizer.json",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": None,
}
GPT_NEO_TOKENIZER = {
    "filename": f"{TEST_DIR}/data/tokenizer/gpt-neo-test-tokenizer.json",
    "bos_token_id": None,
    "eos_token_id": 50279,
    "pad_token_id": 1,
}

TEXT_WITH_NO_NEWLINES = {
    "text": "This is a document with no newlines.",
    "llama": [1, 910, 338, 263, 1842, 411, 694, 716, 9012, 29889, 2],
    "gpt_neo": [1552, 310, 247, 3389, 342, 642, 747, 8737, 15, 50279],
}

TEXT_WITH_NEW_LINES = {
    "text": "A doc with\nnewlines.\n\nToks be the same!\n",
    "llama": [1, 319, 1574, 411, 13, 1482, 9012, 29889, 13, 13, 29911, 12117, 367, 278, 1021, 29991, 13, 2],
    "gpt_neo": [34, 5474, 342, 187, 1826, 8737, 15, 187, 187, 53, 24560, 320, 253, 1072, 2, 187, 50279],
}

TEXT_NEWLINE_START = {
    "text": "\nSimple doc with leading newline.",
    "llama": [1, 29871, 13, 15427, 1574, 411, 8236, 25899, 29889, 2],
    "gpt_neo": [187, 21595, 5474, 342, 4283, 747, 1282, 15, 50279],
}


class TestTokenizer(TestCase):
    def test_llama_process_by_paragraph(self):
        no_split_tok = Tokenizer.from_file(**LLAMA_TOKENIZER, segment_before_tokenization=False)
        split_tok = Tokenizer.from_file(**LLAMA_TOKENIZER, segment_before_tokenization=True)

        # Test that the tokenization is the when segment_before_tokenization is True or False
        # and no newlines are present; we compare with reference tokenization as well
        no_split_tokens = no_split_tok.encode(TEXT_WITH_NO_NEWLINES["text"])
        split_tokens = split_tok.encode(TEXT_WITH_NO_NEWLINES["text"])
        self.assertEqual(no_split_tokens, split_tokens)
        self.assertEqual(split_tokens, TEXT_WITH_NO_NEWLINES["llama"])

        # Test that the tokenization is the same when segment_before_tokenization is True or False
        # and the document has newlines; we compare with reference tokenization as well
        no_split_tokens = no_split_tok.encode(TEXT_WITH_NEW_LINES["text"])
        split_tokens = split_tok.encode(TEXT_WITH_NEW_LINES["text"])
        self.assertEqual(no_split_tokens, split_tokens)
        self.assertEqual(split_tokens, TEXT_WITH_NEW_LINES["llama"])

        # Test how it behaves when the document starts with a newline
        no_split_tokens = no_split_tok.encode(TEXT_NEWLINE_START["text"])
        split_tokens = split_tok.encode(TEXT_NEWLINE_START["text"])
        self.assertEqual(no_split_tokens, split_tokens)
        self.assertEqual(split_tokens, TEXT_NEWLINE_START["llama"])

    def test_gpt_neo_process_by_paragraph(self):
        no_split_tok = Tokenizer.from_file(**GPT_NEO_TOKENIZER, segment_before_tokenization=False)
        split_tok = Tokenizer.from_file(**GPT_NEO_TOKENIZER, segment_before_tokenization=True)

        # Test that the tokenization is the when segment_before_tokenization is True or False
        # and no newlines are present; we compare with reference tokenization as well
        no_split_tokens = no_split_tok.encode(TEXT_WITH_NO_NEWLINES["text"])
        split_tokens = split_tok.encode(TEXT_WITH_NO_NEWLINES["text"])
        self.assertEqual(no_split_tokens, split_tokens)
        self.assertEqual(split_tokens, TEXT_WITH_NO_NEWLINES["gpt_neo"])

        # Test that the tokenization is the same when segment_before_tokenization is True or False
        # and the document has newlines; we compare with reference tokenization as well
        no_split_tokens = no_split_tok.encode(TEXT_WITH_NEW_LINES["text"])
        split_tokens = split_tok.encode(TEXT_WITH_NEW_LINES["text"])
        self.assertEqual(no_split_tokens, split_tokens)
        self.assertEqual(split_tokens, TEXT_WITH_NEW_LINES["gpt_neo"])

        # Test how it behaves when the document starts with a newline
        no_split_tokens = no_split_tok.encode(TEXT_NEWLINE_START["text"])
        split_tokens = split_tok.encode(TEXT_NEWLINE_START["text"])
        self.assertEqual(no_split_tokens, split_tokens)
        self.assertEqual(split_tokens, TEXT_NEWLINE_START["gpt_neo"])


class TestTokenizerCli(TestCase):
    def test_llama_segment_e2e(self):
        config = {
            "destination": f"{TEST_DIR}/work/tokenizer/llama-segment",
            "documents": [
                f"{TEST_DIR}/data/provided/documents/000.json.gz",
            ],
            "processes": 1,
            "seed": 3920,
            "tokenizer": {
                "name_or_path": LLAMA_TOKENIZER["filename"],
                "bos_token_id": LLAMA_TOKENIZER["bos_token_id"],
                "eos_token_id": LLAMA_TOKENIZER["eos_token_id"],
                "pad_token_id": LLAMA_TOKENIZER["pad_token_id"],
                "segment_before_tokenization": True,
            },
            "debug": True,
        }
        tokenizer = BaseTokenizer.from_file(LLAMA_TOKENIZER["filename"])

        with NamedTemporaryFile(mode="wt") as f:
            json.dump(config, f)
            f.flush()
            main(argv=["-c", f.name, "tokens"])

        with smart_open.open(f"{config['destination']}/part-0-00000.csv.gz") as f:
            reader = csv.reader(f)
            metadata = [
                {"start": int(row[0]), "end": int(row[1]), "id": row[2], "src": row[3], "pos": int(row[4])}
                for row in reader
            ]

        size = max(m["end"] for m in metadata)
        memmap = numpy.memmap(
            f"{config['destination']}/part-0-00000.npy", dtype=numpy.uint16, mode="r", shape=(size,)
        )

        with smart_open.open(f"{TEST_DIR}/data/provided/documents/000.json.gz") as f:
            documents = [json.loads(line) for line in f]

        for doc_metadata in metadata:
            original_text = documents[doc_metadata["pos"] - 1]["text"]
            tokens = memmap[doc_metadata["start"] : doc_metadata["end"]]
            tokenized_text = tokenizer.decode(tokens)

            self.assertEqual(tokens[0], LLAMA_TOKENIZER["bos_token_id"])
            self.assertEqual(tokens[-1], LLAMA_TOKENIZER["eos_token_id"])
            self.assertEqual(tokenized_text, original_text)

            # count the number of special tokens in tokens
            special_tokens = sum(
                1
                for t in tokens
                if t
                in {
                    LLAMA_TOKENIZER["bos_token_id"],
                    LLAMA_TOKENIZER["eos_token_id"],
                    LLAMA_TOKENIZER["pad_token_id"],
                }
            )
            self.assertEqual(special_tokens, 2)

    def test_gpt_neo_e2e(self):
        config = {
            "destination": f"{TEST_DIR}/work/tokenizer/gpt-neo-segment",
            "documents": [
                f"{TEST_DIR}/data/provided/documents/000.json.gz",
            ],
            "processes": 1,
            "seed": 3920,
            "tokenizer": {
                "name_or_path": GPT_NEO_TOKENIZER["filename"],
                "bos_token_id": GPT_NEO_TOKENIZER["bos_token_id"],
                "eos_token_id": GPT_NEO_TOKENIZER["eos_token_id"],
                "pad_token_id": GPT_NEO_TOKENIZER["pad_token_id"],
                "segment_before_tokenization": True,
            },
            "debug": True,
        }
        tokenizer = BaseTokenizer.from_file(GPT_NEO_TOKENIZER["filename"])

        with NamedTemporaryFile(mode="wt") as f:
            json.dump(config, f)
            f.flush()
            main(argv=["-c", f.name, "tokens"])

        with smart_open.open(f"{config['destination']}/part-0-00000.csv.gz") as f:
            reader = csv.reader(f)
            metadata = [
                {"start": int(row[0]), "end": int(row[1]), "id": row[2], "src": row[3], "pos": int(row[4])}
                for row in reader
            ]

        size = max(m["end"] for m in metadata)
        memmap = numpy.memmap(
            f"{config['destination']}/part-0-00000.npy", dtype=numpy.uint16, mode="r", shape=(size,)
        )

        with smart_open.open(f"{TEST_DIR}/data/provided/documents/000.json.gz") as f:
            documents = [json.loads(line) for line in f]

        for doc_metadata in metadata:
            original_text = documents[doc_metadata["pos"] - 1]["text"]
            tokens = memmap[doc_metadata["start"] : doc_metadata["end"]]
            tokenized_text = tokenizer.decode(tokens)

            self.assertEqual(tokens[-1], GPT_NEO_TOKENIZER["eos_token_id"])
            self.assertEqual(tokenized_text, original_text)

            # count the number of special tokens in tokens
            special_tokens = sum(
                1
                for t in tokens
                if t
                in {
                    GPT_NEO_TOKENIZER["eos_token_id"],
                    GPT_NEO_TOKENIZER["pad_token_id"],
                }
            )
            self.assertEqual(special_tokens, 1)
