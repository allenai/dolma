from pathlib import Path
from unittest import TestCase

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
