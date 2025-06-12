import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import smart_open

from dolma.cli.__main__ import main as cli_main
from dolma.tokenizer.reshard import ReshardingConfig, reshard

DOLMA2_TOKENIZER = Path(__file__).parent.parent / "data" / "tokenizer" / "dolma2-test-tokenizer.json"


class BaseTestResharding(unittest.TestCase):
    def _read_all_sequences(self, dir: Path) -> tuple[dict[str, list[int]], dict[str, tuple[str, int]]]:
        sequences: dict[str, list[int]] = {}
        locations: dict[str, tuple[str, int]] = {}
        # check if contents are the same. order might be different so make you have to fetch metadata and compare
        for npy_path in dir.rglob("*.npy"):
            arr = np.memmap(npy_path, mode="r", dtype=np.uint32, shape=(npy_path.stat().st_size // 4,))
            csv_path = npy_path.with_suffix(".csv.gz")
            with smart_open.open(csv_path, "r") as f:
                for row in csv.reader(f):
                    start, end, seq_id, path, loc = row
                    sequences[seq_id] = arr[int(start) : int(end)].tolist()
                    locations[seq_id] = (path, int(loc))
        return sequences, locations

    def _check_equal(self, input_dir: Path, output_dir: Path):
        # check if size of npy files in input dir is less than target size
        input_dir_size = sum(f.stat().st_size for f in input_dir.rglob("*.npy"))
        output_dir_size = sum(f.stat().st_size for f in output_dir.rglob("*.npy"))
        self.assertEqual(input_dir_size, output_dir_size)

        print(f"Found {input_dir_size} bytes in {input_dir}")
        print(f"Found {output_dir_size} bytes in {output_dir}")

        input_csv_rows = 0
        # check csv files have same number of rows in input and output dir
        for path in input_dir.rglob("*.csv.gz"):
            with smart_open.open(path, "r") as f:
                input_csv_rows += sum(1 for _ in f)

        output_csv_rows = 0
        for path in output_dir.rglob("*.csv.gz"):
            with smart_open.open(path, "r") as f:
                output_csv_rows += sum(1 for _ in f)
        self.assertEqual(input_csv_rows, output_csv_rows)
        print(f"Found {input_csv_rows} csv rows in {input_dir}")
        print(f"Found {output_csv_rows} csv rows in {output_dir}")

        input_sequences, input_locations = self._read_all_sequences(input_dir)
        output_sequences, output_locations = self._read_all_sequences(output_dir)

        print(f"Found {len(input_sequences)} sequences in {input_dir}")
        print(f"Found {len(output_sequences)} sequences in {output_dir}")

        for k in input_sequences:
            self.assertIn(k, output_sequences)
            self.assertEqual(input_sequences[k], output_sequences[k])

        for k in input_locations:
            self.assertIn(k, output_locations)
            self.assertEqual(input_locations[k], output_locations[k])


class TestSynthResharding(BaseTestResharding):

    def make_memmap_and_csv(
        self,
        path: str,
        doc_contents: list[list[int]],
        doc_name: str,
    ):
        csv_path = Path(path).with_suffix(".csv.gz")
        npy_path = Path(path).with_suffix(".npy")

        npy_path.parent.mkdir(parents=True, exist_ok=True)

        with smart_open.open(csv_path, "w") as f:
            writer = csv.writer(f)
            total_length = sum(len(seq) for seq in doc_contents)
            memmap_file = np.memmap(npy_path, mode="w+", shape=(total_length,), dtype=np.uint32)

            current_offset = 0
            for i, seq in enumerate(doc_contents):
                memmap_file[current_offset : current_offset + len(seq)] = seq
                writer.writerow([current_offset, current_offset + len(seq), f"{doc_name}_s{i:03d}", path, i])
                current_offset += len(seq)
            memmap_file.flush()

    def setUp(self):
        np.random.seed(42)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_resharding(self):
        target_size = 1024  # 1 KB, doesnt matter

        doc1_contents = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ]
        doc2_contents = [
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [31, 32],
            [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
            [51, 52, 53, 54, 55],
            [56, 57, 58, 59, 60],
            [61, 62, 63, 64, 65, 66, 67, 68, 69],
            [70],
        ]
        for i, doc_contents in enumerate([doc1_contents, doc2_contents]):
            self.make_memmap_and_csv(
                path=f"{self.tmp_dir}/input/{i:03d}",
                doc_contents=doc_contents,
                doc_name=f"d{i:03d}",
            )

        input_dir = Path(self.tmp_dir) / "input"
        output_dir = Path(self.tmp_dir) / "output"

        config = ReshardingConfig.from_dict(
            {
                "source_prefixes": [
                    {
                        "prefix": str(input_dir),
                        "sample_rate": 1.0,
                    }
                ],
                "destination_prefix": str(output_dir),
                "max_size_bytes": target_size,
                "random_seed": 42,
                "tokenizer_name_or_path": str(DOLMA2_TOKENIZER),
            }
        )
        reshard(config)

        # check if size of npy files in input dir is less than target size
        input_dir_size = sum(f.stat().st_size for f in input_dir.rglob("*.npy"))
        output_dir_size = sum(f.stat().st_size for f in output_dir.rglob("*.npy"))
        self.assertEqual(input_dir_size, output_dir_size)
        self._check_equal(input_dir, output_dir)


class TestEndToEndResharding(BaseTestResharding):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _make_documents(self, texts: list[str], doc_name: str, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with smart_open.open(path, "w") as f:
            for i, text in enumerate(texts):
                doc = {"id": f"{doc_name}_s{i:03d}", "text": text}
                f.write(json.dumps(doc) + "\n")

    def test_end_to_end_resharding(self):
        docs = [
            [
                [
                    "Hi!",
                    "The quantum computer calculated seventeen billion possibilities in 0.003 seconds, which honestly made everyone question reality.",
                    "Socks disappear mysteriously.",
                ],
                [
                    "Why do rubber ducks make debugging so much easier than talking to actual humans?",
                    "Pizza exists.",
                    "Her grandmother's antique typewriter still clicked with the persistence of forgotten love letters.",
                    "Oof.",
                    "The blockchain revolutionizes trust mechanisms through distributed consensus algorithms.",
                    "Dancing happens.",
                    "When philosophers contemplate the nature of existence, do they ever wonder if existence contemplates them back?",
                ],
            ],
            [
                ["Meow."],
                [
                    "The WiFi password was 'password123' and nobody felt secure about it.",
                    "Oxygen molecules are having a party in your lungs right now!",
                    "Chairs support dreams.",
                ],
                [
                    "ERROR 404: Motivation not found, please try again after coffee.",
                    "Photosynthesis rocks!",
                    "The universe expanded another few micrometers while you read this sentence, making everything slightly more spacious.",
                    "Bananas curve naturally.",
                    "JSON syntax errors cause more existential crises than actual philosophy classes.",
                ],
                [
                    "Wow!",
                    "Memory foam remembers everything except where you put your keys.",
                    "Artificial intelligence learned to play chess but still can't figure out why humans cry at movies.",
                ],
            ],
            [
                [
                    "Gravity works.",
                    "The barista drew a heart in the latte foam, and suddenly the Monday morning felt like a warm hug from the universe.",
                    "Algorithms dream of electric sheep.",
                    "Clouds float because they believe in themselves.",
                    "SQL queries sometimes feel like poetry written in a very practical language.",
                    "Batteries die unexpectedly.",
                    "The sunset painted the sky in colors that don't have names yet, as if nature was showing off its artistic license.",
                ]
            ],
            [
                [
                    "Hmm.",
                    "Machine learning models trained on cat videos probably understand feline behavior better than most veterinarians.",
                ],
                [
                    "The paper airplane defied physics and landed exactly where intended, proving that hope sometimes has aerodynamic properties."
                ],
                [
                    "APIs communicate.",
                    "Toast lands butter-side down because the universe has a sense of humor about breakfast mishaps.",
                    "Debugging is like being a detective in a crime movie where you are also the murderer.",
                    "Mathematics whispers secrets.",
                ],
                [
                    "Recursive functions call themselves until someone tells them to stop, much like existential thoughts at 3 AM.",
                    "Nope.",
                    "The refrigerator hummed a gentle tune that only the midnight snack seekers could truly appreciate.",
                    "Variables store hopes.",
                    "Encryption protects secrets the way diary locks protected teenage thoughts in the 1990s.",
                    "Time zones confuse everyone.",
                    "The rubber plant in the corner judged everyone's programming skills silently but fairly.",
                    "Cache invalidation remains one of the hardest problems in computer science and in life generally.",
                ],
            ],
        ]

        doc_groups = []
        for i, doc_group in enumerate(docs):
            group_path = self.tmp_dir / "documents" / f"g{i:03d}"
            doc_groups.append(group_path)

            for j, doc_texts in enumerate(doc_group):
                self._make_documents(
                    texts=doc_texts * 10,
                    doc_name=f"d{i:03d}_f{j:03d}",
                    path=group_path / f"f{j:03d}.jsonl",
                )

        for group_path in doc_groups:
            tokenizer_config = {
                "documents": [f"{group_path}/*.jsonl"],
                "destination": f"{self.tmp_dir}/tokens/{group_path.name}",
                "tokenizer": {
                    "name_or_path": str(DOLMA2_TOKENIZER),
                    "eos_token_id": 100257,
                    "pad_token_id": 100277,
                },
                "max_size": 100,
                "debug": True,
                "dtype": "uint32",
            }

            with tempfile.NamedTemporaryFile(mode="w") as f:
                json.dump(tokenizer_config, f)
                f.flush()
                cli_main(["-c", f.name, "tokens"])

        config = ReshardingConfig.from_dict(
            {
                "source_prefixes": [
                    {"prefix": f"{self.tmp_dir}/tokens", "sample_rate": 1.0},
                ],
                "destination_prefix": f"{self.tmp_dir}/resharded",
                "max_size_bytes": 1000,
                "random_seed": 42,
                "tokenizer_name_or_path": str(DOLMA2_TOKENIZER),
            }
        )
        reshard(config)

        self._check_equal(Path(self.tmp_dir) / "tokens", Path(self.tmp_dir) / "resharded")
