import copy
import csv
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp
from typing import Optional
from unittest import TestCase

import numpy
import smart_open
from tokenizers import Tokenizer as BaseTokenizer
from typing_extensions import TypedDict

from dolma.cli.__main__ import main
from dolma.tokenizer import Tokenizer, tokenize_in_parallel

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
LLAMA3_TOKENIZER = {
    "filename": f"{TEST_DIR}/data/tokenizer/llama3-test-tokenizer.json",
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "pad_token_id": None,
}
DOLMA2_TOKENIZER = {
    "filename": f"{TEST_DIR}/data/tokenizer/dolma2-test-tokenizer.json",
    "bos_token_id": None,
    "eos_token_id": 100257,
    "pad_token_id": 100277,
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


class MetadataDict(TypedDict):
    start: int
    end: int
    id: str
    src: str
    pos: int


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
    def test_llama_segment_e2e(self, segment: bool = True, fast: bool = True, refresh: int = 0):
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
                "segment_before_tokenization": segment,
                "refresh": refresh,
                "fast": fast,
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
                MetadataDict(start=int(row[0]), end=int(row[1]), id=row[2], src=row[3], pos=int(row[4]))
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

    def test_gpt_neo_e2e(self, segment: bool = True, fast: bool = True, refresh: int = 0):
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
                "segment_before_tokenization": segment,
                "refresh": refresh,
                "fast": fast,
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
                MetadataDict(start=int(row[0]), end=int(row[1]), id=row[2], src=row[3], pos=int(row[4]))
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

    def test_llama3_e2e(self, segment: bool = True, fast: bool = True, refresh: int = 0):
        config = {
            "destination": f"{TEST_DIR}/work/tokenizer/gpt-neo-segment",
            "documents": [
                f"{TEST_DIR}/data/provided/documents/000.json.gz",
            ],
            "processes": 1,
            "seed": 3920,
            "dtype": "uint32",
            "tokenizer": {
                "name_or_path": LLAMA3_TOKENIZER["filename"],
                "bos_token_id": LLAMA3_TOKENIZER["bos_token_id"],
                "eos_token_id": LLAMA3_TOKENIZER["eos_token_id"],
                "pad_token_id": LLAMA3_TOKENIZER["pad_token_id"],
                "segment_before_tokenization": segment,
                "refresh": refresh,
                "fast": fast,
            },
            "debug": True,
        }
        tokenizer = BaseTokenizer.from_file(LLAMA3_TOKENIZER["filename"])

        with NamedTemporaryFile(mode="wt") as f:
            json.dump(config, f)
            f.flush()
            main(argv=["-c", f.name, "tokens"])

        with smart_open.open(f"{config['destination']}/part-0-00000.csv.gz") as f:
            reader = csv.reader(f)
            metadata = [
                MetadataDict(start=int(row[0]), end=int(row[1]), id=row[2], src=row[3], pos=int(row[4]))
                for row in reader
            ]

        size = max(m["end"] for m in metadata)
        memmap = numpy.memmap(
            f"{config['destination']}/part-0-00000.npy", dtype=numpy.uint32, mode="r", shape=(size,)
        )

        with smart_open.open(f"{TEST_DIR}/data/provided/documents/000.json.gz") as f:
            documents = [json.loads(line) for line in f]

        for doc_metadata in metadata:
            original_text = documents[doc_metadata["pos"] - 1]["text"]
            tokens = memmap[doc_metadata["start"] : doc_metadata["end"]]
            tokenized_text = tokenizer.decode(tokens)

            self.assertEqual(tokens[-1], LLAMA3_TOKENIZER["eos_token_id"])
            self.assertEqual(tokenized_text, original_text)

            # count the number of special tokens in tokens
            special_tokens = sum(
                1
                for t in tokens
                if t
                in {
                    LLAMA3_TOKENIZER["eos_token_id"],
                    LLAMA3_TOKENIZER["pad_token_id"],
                }
            )
            self.assertEqual(special_tokens, 1)

    def test_llama_segment_e2e_no_segment(self):
        self.test_llama_segment_e2e(segment=False)

    def test_gpt_neo_segment_e2e_no_segment(self):
        self.test_gpt_neo_e2e(segment=False)

    def test_llama_segment_e2e_refresh(self):
        self.test_llama_segment_e2e(refresh=1)

    def test_gpt_neo_segment_e2e_refresh(self):
        self.test_gpt_neo_e2e(refresh=1)


class TestShufflingTokenizer(TestCase):
    def test_shuffling(self):
        tokenizer_id = "allenai/olmo-1b"
        bos_token_id = None
        eos_token_id = 50279
        pad_token_id = 1

        tokenizer = Tokenizer.from_pretrained(
            tokenizer_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id
        )

        tmpdir = Path(mkdtemp())
        try:
            (source := tmpdir / "src").mkdir(parents=True, exist_ok=True)
            (destination := tmpdir / "dst").mkdir(parents=True, exist_ok=True)

            RING_SIZE = 4
            LOCAL_SHUFFLE = 8
            REPEAT = 4

            for i in range(RING_SIZE):
                with smart_open.open(source / f"{i}.jsonl.gz", "wt") as f:
                    for j in range(LOCAL_SHUFFLE * REPEAT):
                        f.write(json.dumps({"text": str(j), "id": f"{i}-{j}"}) + "\n")

            tokenize_in_parallel(
                sources=[f"{source}/*.gz"],
                destination=str(destination),
                ring_size=RING_SIZE,
                local_shuffle=LOCAL_SHUFFLE,
                tokenizer_name_or_path=tokenizer_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                seed=3920,
                debug=False,
                num_writers=1,
            )

            with smart_open.open(destination / "part-0-00000.csv.gz") as f:
                reader = csv.reader(f)
                file_ids, row_data = zip(*(map(int, row[2].split("-")) for row in reader))

            memmap = numpy.memmap(
                f"{destination}/part-0-00000.npy",
                dtype=numpy.uint16,
                mode="r",
                # * 2 if because we have eos tokens
                shape=(RING_SIZE * LOCAL_SHUFFLE * REPEAT * 2,),
            )

            # decode the numpy here. We decode one document at the time, i.e. until we see the eos token
            all_tokens, current_tokens = [], []
            for elem in memmap:
                if elem == eos_token_id:
                    all_tokens.append(int(tokenizer.decode(current_tokens)))
                    current_tokens = []
                else:
                    current_tokens.append(elem)

            if current_tokens:
                all_tokens.append(tokenizer.decode(current_tokens))

            # verify that the correct number of tokens have been written
            self.assertEqual(all_tokens, list(row_data))
            self.assertEqual(len(all_tokens), RING_SIZE * LOCAL_SHUFFLE * REPEAT)

            # verify that there has bee shuffling
            self.assertNotEqual(list(all_tokens), sorted(all_tokens))

        finally:
            shutil.rmtree(tmpdir)


class TestTokenizeSpecialTokens(TestCase):
    def test_tokenize_special_tokens(self):
        tokenizer_default = Tokenizer.from_file(**DOLMA2_TOKENIZER)
        tokenizer_split = Tokenizer.from_file(**DOLMA2_TOKENIZER, encode_special_tokens=True)

        text = "This is a test document."
        tokens_default = tokenizer_default.encode(text)
        tokens_split = tokenizer_split.encode(text)
        self.assertEqual(tokens_default, tokens_split)

        text = "This document explains what <|endoftext|> is."
        tokens_default = tokenizer_default.encode(text)
        tokens_split = tokenizer_split.encode(text)
        self.assertNotEqual(tokens_default, tokens_split)
        self.assertEqual(
            tokenizer_default.decode(tokens_default, skip_special_tokens=True), "This document explains what  is."
        )
        self.assertEqual(tokenizer_split.decode(tokens_split, skip_special_tokens=True), text)
        self.assertEqual(
            tokenizer_default.decode(tokens_default, skip_special_tokens=False),
            tokenizer_split.decode(tokens_split, skip_special_tokens=False),
        )

        text = "This document explains contain a |||PHONE_NUMBER||| number."
        tokens_default = tokenizer_default.encode(text)
        tokens_split = tokenizer_split.encode(text)
        self.assertEqual(tokens_default, tokens_split)


class TestBosEosTokenAddition(TestCase):
    def setUp(self):
        self.tokenizer_config = DOLMA2_TOKENIZER
        self.documents = [
            "I do not like living barely on the edge of legality.",
            "Poor is the man who does not see the sun's rays.",
            "Ishmishing is a word that is not in the dictionary, but in our hearts.",
            "All must dress for the main event.",
            "Coded springs are behind us.",
        ]

        self.tokenizer = Tokenizer.from_file(**self.tokenizer_config)

        self.document_tokenized = [self.tokenizer.encode(doc, add_special_tokens=False) for doc in self.documents]

        self.tmpdir = Path(mkdtemp())
        self.input_dir = self.tmpdir / "input"
        self.output_dir = self.tmpdir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with smart_open.open(self.input_dir / "000.json.gz", "wt") as f:
            for i, doc in enumerate(self.documents):
                f.write(json.dumps({"text": doc, "id": f"doc-{i}"}) + "\n")

        self.default_config = {
            "destination": f"{self.output_dir}",
            "documents": [f"{self.input_dir}/*.json.gz"],
            "processes": 1,
            "seed": 3920,
            "dtype": "uint32",
        }

    def tearDown(self):
        """Clean up any resources created in setUp."""
        if hasattr(self, "tmpdir"):
            shutil.rmtree(self.tmpdir)

    def _get_config(self, add_bos: bool = False, add_eos: bool = False):
        config = copy.deepcopy(self.default_config)
        config["tokenizer"] = {
            "name_or_path": self.tokenizer_config["filename"],
            "pad_token_id": self.tokenizer_config["pad_token_id"],
        }
        if add_bos:
            config["tokenizer"]["bos_token_id"] = self.tokenizer_config[
                "eos_token_id"
            ]  # using eos token as bos token
        if add_eos:
            config["tokenizer"]["eos_token_id"] = self.tokenizer_config["eos_token_id"]
        return config

    def _run_tokenizer_and_read_output(self, config: dict) -> tuple[list[MetadataDict], list[int]]:
        with NamedTemporaryFile(mode="wt") as f:
            json.dump(config, f)
            f.flush()
            main(argv=["-c", f.name, "tokens"])

        with smart_open.open(Path(config["destination"]) / "part-0-00000.csv.gz", "rt", encoding="utf-8") as f:
            reader = csv.reader(f)
            metadata = [
                MetadataDict(start=int(row[0]), end=int(row[1]), id=row[2], src=row[3], pos=int(row[4]))
                for row in reader
            ]

        with smart_open.open(Path(config["destination"]) / "part-0-00000.npy", "rb") as f:
            contents = numpy.memmap(f, dtype=config["dtype"], mode="r").tolist()

        return metadata, contents

    def test_adding_bos_token(self):
        config = self._get_config(add_bos=True)
        _, contents = self._run_tokenizer_and_read_output(config)

        # we should have as many bos tokens as documents
        count_sep = sum(1 for t in contents if t == self.tokenizer_config["eos_token_id"])
        self.assertEqual(count_sep, len(self.documents))

        # the sequence should start with a bos token
        self.assertEqual(contents[0], self.tokenizer_config["eos_token_id"])

        # the sequence should NOT end with an eos token
        self.assertNotEqual(contents[-1], self.tokenizer_config["eos_token_id"])

        # partition sequences
        extracted_sequences: list[list[int]] = []
        for i, token in enumerate(contents):
            if i == 0:
                extracted_sequences.append([])
            elif token != self.tokenizer_config["eos_token_id"]:
                extracted_sequences[-1].append(token)
            else:
                extracted_sequences.append([])

        # now check if partitioned sequences exist in the original document tokenized
        # note that order can be different cuz we shuffle during tokenization
        expected = self.document_tokenized[:]
        for sequence in extracted_sequences:
            pos = expected.index(sequence)
            self.assertEqual(sequence, expected[pos])
            expected.pop(pos)
        self.assertEqual(expected, [])

    def test_adding_eos_token(self):
        config = self._get_config(add_eos=True)
        _, contents = self._run_tokenizer_and_read_output(config)

        # we should have as many eos tokens as documents
        count_sep = sum(1 for t in contents if t == self.tokenizer_config["eos_token_id"])
        self.assertEqual(count_sep, len(self.documents))

        # the sequence should end with an eos token
        self.assertEqual(contents[-1], self.tokenizer_config["eos_token_id"])

        # the sequence should NOT start with a bos token
        self.assertNotEqual(contents[0], self.tokenizer_config["eos_token_id"])

        # partition sequences
        extracted_sequences = [[]]
        for i, token in enumerate(contents):
            if i == len(contents) - 1:
                # do nothing on last symbol (should be eos, but you checked that above)
                continue
            elif token != self.tokenizer_config["eos_token_id"]:
                extracted_sequences[-1].append(token)
            else:
                extracted_sequences.append([])

        # now check if partitioned sequences exist in the original document tokenized
        # note that order can be different cuz we shuffle during tokenization
        expected = self.document_tokenized[:]
        for sequence in extracted_sequences:
            pos = expected.index(sequence)
            self.assertEqual(sequence, expected[pos])
            expected.pop(pos)
        self.assertEqual(expected, [])

    def test_adding_bos_and_eos_tokens(self):
        config = self._get_config(add_bos=True, add_eos=True)
        _, contents = self._run_tokenizer_and_read_output(config)

        # we should have as many bos/eos tokens as twice the documents (we use same symbol for bos and eos)
        count_sep = sum(1 for t in contents if t == self.tokenizer_config["eos_token_id"])
        self.assertEqual(count_sep, len(self.documents) * 2)

        # the sequence should start with a bos token
        self.assertEqual(contents[0], self.tokenizer_config["eos_token_id"])

        # the sequence should end with an eos token
        self.assertEqual(contents[-1], self.tokenizer_config["eos_token_id"])

        # partition sequences
        extracted_sequences = []
        for i, token in enumerate(contents):
            if i == 0:
                # first bos
                extracted_sequences.append([])
            if i == len(contents) - 1:
                # do nothing on last symbol (should be eos, but you checked that above)
                continue
            elif token != self.tokenizer_config["eos_token_id"]:
                extracted_sequences[-1].append(token)
            elif contents[i - 1] != self.tokenizer_config["eos_token_id"]:
                # this is EOS, otherwise previous token wouldnt be EOS
                extracted_sequences.append([])

        # now check if partitioned sequences exist in the original document tokenized
        # note that order can be different cuz we shuffle during tokenization
        expected = self.document_tokenized[:]
        for sequence in extracted_sequences:
            pos = expected.index(sequence)
            self.assertEqual(sequence, expected[pos])
            expected.pop(pos)
        self.assertEqual(expected, [])


class TokenizeOnNonStandardFields(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.default_config = {
            "tokenizer": {
                "name_or_path": DOLMA2_TOKENIZER["filename"],
                "bos_token_id": DOLMA2_TOKENIZER["eos_token_id"],
                "eos_token_id": DOLMA2_TOKENIZER["eos_token_id"],
                "pad_token_id": DOLMA2_TOKENIZER["pad_token_id"],
            },
            "processes": 1,
            "seed": 3920,
            "dtype": "uint32",
            "debug": True,
        }
        self.texts = sorted(
            [
                "To be or not to be, that is the question.",
                "I think, therefore I am.",
                "The only way to do great work is to love what you do.",
                "Be the change you wish to see in the world.",
                "All that glitters is not gold.",
                "The road not taken makes all the difference.",
                "To thine own self be true.",
                "Life is what happens while you're busy making other plans.",
                "The early bird catches the worm.",
                "Where there's a will, there's a way.",
            ]
        )
        self.tokenizer = Tokenizer.from_file(**DOLMA2_TOKENIZER)

    def tearDown(self):
        """Clean up any resources created in setUp."""
        self.stack.close()

    def _make_documents(self, text_field: str, id_field: Optional[str]):
        def make_text_doc(my_text_field: str, my_text_content: str):
            if "." in my_text_field:
                prefix, rest = my_text_field.split(".", 1)
                return {prefix: make_text_doc(rest, my_text_content)}
            else:
                return {my_text_field: my_text_content}

        def make_id_doc(my_id_field: Optional[str], my_id_content: str):
            return {} if my_id_field is None else make_text_doc(my_id_field, my_id_content)

        input_dir = self.stack.enter_context(TemporaryDirectory())
        output_dir = self.stack.enter_context(TemporaryDirectory())

        for i, text in enumerate(self.texts):
            with smart_open.open(f"{input_dir}/{i:03d}.json.gz", "wt", encoding="utf-8") as f:
                doc = {**make_text_doc(text_field, text), **make_id_doc(id_field, f"doc-{i}")}
                f.write(json.dumps(doc) + "\n")

        return input_dir, output_dir

    def _run_tokenizer_and_read_output(self, config: dict) -> list[int]:
        with NamedTemporaryFile(mode="wt") as f:
            json.dump(config, f)
            f.flush()
            main(argv=["-c", f.name, "tokens"])

        contents = []
        for fn in Path(config["destination"]).glob("*.npy"):
            with smart_open.open(fn, "rb") as f:
                contents.extend(numpy.memmap(f, dtype=config["dtype"], mode="r").tolist())
        return contents

    def _decode_contents(self, contents: list[int]) -> list[str]:
        # partition sequences
        extracted_sequences: list[list[int]] = []
        for i, token in enumerate(contents):
            if i == 0:
                # first bos
                extracted_sequences.append([])
            if i == len(contents) - 1:
                # do nothing on last symbol (should be eos, but you checked that above)
                continue
            elif token != DOLMA2_TOKENIZER["eos_token_id"]:
                extracted_sequences[-1].append(token)
            elif contents[i - 1] != DOLMA2_TOKENIZER["eos_token_id"]:
                # this is EOS, otherwise previous token wouldnt be EOS
                extracted_sequences.append([])

        return sorted([self.tokenizer.decode(seq) for seq in extracted_sequences])

    def test_tokenize_with_no_id_field(self):
        input_dir, output_dir = self._make_documents(text_field="text", id_field=None)
        config = copy.deepcopy(self.default_config)
        config["documents"] = [f"{input_dir}/*.json.gz"]
        config["destination"] = output_dir
        config.setdefault("fields", {})["id_field_name"] = None
        contents = self._run_tokenizer_and_read_output(config)
        decoded = self._decode_contents(contents)

        for src, dst in zip(self.texts, decoded):
            self.assertEqual(src, dst)

    def test_with_nested_text_and_id_field(self):
        input_dir, output_dir = self._make_documents(text_field="text.nested", id_field="id.nested.more")
        config = copy.deepcopy(self.default_config)
        config["documents"] = [f"{input_dir}/*.json.gz"]
        config["destination"] = output_dir
        config.setdefault("fields", {})["id_field_name"] = "id.nested.more"
        config.setdefault("fields", {})["text_field_name"] = "text.nested"
        contents = self._run_tokenizer_and_read_output(config)
        decoded = self._decode_contents(contents)

        for src, dst in zip(self.texts, decoded):
            self.assertEqual(src, dst)


class TestDtypeMismatch(TestCase):
    def test_dtype_mismatch(self):
        with self.assertRaises(TypeError), TemporaryDirectory() as tmpdir:
            tokenize_in_parallel(
                tokenizer_name_or_path=DOLMA2_TOKENIZER["filename"],
                sources=[tmpdir],
                destination=tmpdir,
                dtype="uint16",
            )
