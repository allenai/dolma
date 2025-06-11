import csv
from pathlib import Path
import shutil
import unittest
import tempfile
from dolma.cli.__main__ import main as cli_main
from dolma.tokenizer.reshard import main as reshard_main
import numpy as np
import smart_open


DOLMA2_TOKENIZER = Path(__file__).parent.parent / "data" / "tokenizer" / "dolma2-test-tokenizer.json"


class TestSynthResharding(unittest.TestCase):

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
            memmap_file = np.memmap(npy_path, mode="w+", shape=(total_length, ), dtype=np.uint32)

            current_offset = 0
            for i, seq in enumerate(doc_contents):
                memmap_file[current_offset:current_offset+len(seq)] = seq
                writer.writerow([current_offset, current_offset+len(seq), f"{doc_name}_s{i:03d}", path, i])
                current_offset += len(seq)
            memmap_file.flush()

    def setUp(self):
        np.random.seed(42)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


    def _read_all_sequences(self, dir: Path) -> tuple[dict[str, list[int]], dict[str, tuple[str, int]]]:
        sequences: dict[str, list[int]] = {}
        locations: dict[str, tuple[str, int]] = {}
        # check if contents are the same. order might be different so make you have to fetch metadata and compare
        for npy_path in dir.rglob("*.npy"):
            arr = np.memmap(npy_path, mode="r", dtype=np.uint32, shape=(npy_path.stat().st_size // 4, ))
            csv_path = npy_path.with_suffix(".csv.gz")
            with smart_open.open(csv_path, "r") as f:
                for row in csv.reader(f):
                    start, end, seq_id, path, loc = row
                    sequences[seq_id] = arr[int(start):int(end)].tolist()
                    locations[seq_id] = (path, int(loc))
        return sequences, locations

    def test_resharding(self):
        target_size = 1024 #1 KB, doesnt matter

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
            [70]
        ]
        for i, doc_contents in enumerate([doc1_contents, doc2_contents]):
            self.make_memmap_and_csv(
                path=f"{self.tmp_dir}/input/{i:03d}",
                doc_contents=doc_contents,
                doc_name=f"d{i:03d}",
            )

        input_dir = Path(self.tmp_dir) / "input"
        output_dir = Path(self.tmp_dir) / "output"
        reshard_main(
            source_prefix=str(input_dir),
            destination_prefix=str(output_dir),
            min_size=target_size,
            max_workers=1,
            random_seed=42,
            tokenizer_name_or_path=str(DOLMA2_TOKENIZER),
        )

        # check if size of npy files in input dir is less than target size
        input_dir_size = sum(f.stat().st_size for f in input_dir.rglob("*.npy"))
        output_dir_size = sum(f.stat().st_size for f in output_dir.rglob("*.npy"))
        self.assertEqual(input_dir_size, output_dir_size)

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

        input_sequences, input_locations = self._read_all_sequences(input_dir)
        output_sequences, output_locations = self._read_all_sequences(output_dir)
        for k in input_sequences:
            self.assertIn(k, output_sequences)
            self.assertEqual(input_sequences[k], output_sequences[k])

        for k in input_locations:
            self.assertIn(k, output_locations)
            self.assertEqual(input_locations[k], output_locations[k])
