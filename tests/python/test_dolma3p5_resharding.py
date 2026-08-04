import argparse
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import yaml

from dolma.tokenizer.reshard import (
    ReshardingConfig,
    ReshardingManifestConfig,
    destination_has_objects,
    reshard,
    upload_to_s3,
)
from scripts.dolma3p5_resharding.workflow import (
    PreparationError,
    S3Object,
    _allocate_object_repetitions,
    _exclusive_directory,
    _finalize_inventory,
    _parse_s5cmd_jsonl,
    _partition_object_uses,
    collect_inventory,
    plan_build,
    preflight_build,
    propose_configs,
    validate_build,
    verify_output,
)


class TestDolma35ReshardingPreparation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.mix = self.root / "mix.yaml"
        self.catalog = self.root / "catalog.csv"
        mix = {
            "mix": [
                {
                    "name": "catalog-source:topic",
                    "weight": 0.5,
                    "categories": [
                        {
                            "name": "default",
                            "weight": 1.0,
                            "paths": [
                                "dolma3p5_pool/catalog-source/topic/allenai/tokenizer/*.npy"
                            ],
                            "repetition_factor": -1.0,
                        }
                    ],
                },
                {
                    "name": "the-stack-v2:Tcl",
                    "weight": 0.25,
                    "categories": [
                        {
                            "name": "high",
                            "weight": 1.0,
                            "paths": [
                                "dolma3p5_pool/the-stack-v2/data/Tcl/quality_p95/allenai/tokenizer/*.npy",
                                "dolma3p5_pool/the-stack-v2/data/Tcl/quality_p95/allenai/tokenizer/*.npy",
                            ],
                            "repetition_factor": -1.0,
                        }
                    ],
                },
                {
                    "name": "direct-source:default",
                    "weight": 0.25,
                    "categories": [
                        {
                            "name": "default",
                            "weight": 1.0,
                            "paths": [
                                "preprocessed/direct-source/allenai/tokenizer/*.npy"
                            ],
                            "repetition_factor": -1.0,
                        }
                    ],
                },
            ]
        }
        self.mix.write_text(yaml.safe_dump(mix, sort_keys=False))
        self.catalog.write_text(
            "ai2-llm,preprocessed/catalog-source/topic/allenai/tokenizer/0000.npy\n"
            "ai2-llm,preprocessed/catalog-source/topic/allenai/tokenizer/0000.csv.gz\n"
            "ai2-llm,preprocessed/the-stack-v2/data/Tcl/quality_p95/allenai/tokenizer/0000.npy\n"
        )
        self.build = self.root / "build"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _plan(self):
        plan_build(
            argparse.Namespace(
                mix=self.mix,
                catalog=self.catalog,
                settings=None,
                output=self.build,
            )
        )

    def _write_inventory(self):
        listing = self.root / "listing.jsonl"
        with listing.open("x") as f:
            for uri, size in self._inventory_objects().items():
                f.write(json.dumps(self._inventory_record(uri, size)) + "\n")
        phase = self.build / "02-inventory"
        _exclusive_directory(phase)
        objects, errors = _parse_s5cmd_jsonl(listing)
        self.assertFalse(errors)
        _finalize_inventory(
            self.build,
            phase,
            objects,
            client=MagicMock(),
            max_workers=1,
        )

    @staticmethod
    def _inventory_objects():
        return {
            "s3://ai2-llm/preprocessed/catalog-source/topic/allenai/tokenizer/0000.npy": 400,
            "s3://ai2-llm/preprocessed/catalog-source/topic/allenai/tokenizer/0000.csv.gz": 20,
            "s3://ai2-llm/preprocessed/the-stack-v2/data/Tcl/quality_p95/allenai/tokenizer/0000.npy": 800,
            "s3://ai2-llm/preprocessed/the-stack-v2/data/Tcl/quality_p95/allenai/tokenizer/0000.csv.gz": 24,
            "s3://ai2-llm/preprocessed/direct-source/allenai/tokenizer/0000.npy": 1200,
            "s3://ai2-llm/preprocessed/direct-source/allenai/tokenizer/0000.csv.gz": 28,
        }

    @staticmethod
    def _inventory_record(uri, size):
        return {
            "key": uri,
            "etag": "test-etag",
            "last_modified": "2026-01-01T00:00:00Z",
            "type": "file",
            "size": size,
            "storage_class": "STANDARD",
        }

    def test_inventory_collector_hides_backend_selection(self):
        self._plan()
        captured_command = []

        def run_collector(command, *, stdout, **kwargs):
            captured_command.extend(command)
            for uri, size in self._inventory_objects().items():
                stdout.write(json.dumps(self._inventory_record(uri, size)) + "\n")
            return SimpleNamespace(returncode=0, stderr="")

        session = MagicMock()
        with (
            patch(
                "scripts.dolma3p5_resharding.workflow.shutil.which",
                return_value="/usr/bin/s5cmd",
            ),
            patch(
                "scripts.dolma3p5_resharding.workflow.subprocess.run",
                side_effect=run_collector,
            ),
            patch(
                "scripts.dolma3p5_resharding.workflow.boto3.Session",
                return_value=session,
            ),
        ):
            collect_inventory(
                argparse.Namespace(
                    build=self.build,
                    profile="read-only",
                    region="us-west-2",
                    max_workers=2,
                )
            )

        self.assertEqual(captured_command[0], "s5cmd")
        collector = json.loads((self.build / "02-inventory/collector.json").read_text())
        self.assertEqual(collector["collector"], "s5cmd")
        self.assertTrue((self.build / "02-inventory/raw-listings.jsonl").is_file())

    def test_end_to_end_preparation_is_create_only(self):
        self._plan()
        with (self.build / "01-plan/corrections.csv").open() as f:
            corrections = list(csv.DictReader(f))
        self.assertEqual(len(corrections), 1)
        self.assertIn("quality_p95", corrections[0]["path"])
        with (self.build / "01-plan/direct-s3-patterns.csv").open() as f:
            direct = list(csv.DictReader(f))
        self.assertEqual(len(direct), 1)
        with self.assertRaises(PreparationError):
            self._plan()

        self._write_inventory()
        propose_configs(
            argparse.Namespace(
                build=self.build,
                destination_root="s3://test-bucket/new-datasets/dolma3p5",
                local_temp_root=str(self.root / "temp-base"),
                max_unit_working_bytes=20_000_000_000_000,
            )
        )
        configs = list((self.build / "03-proposal/config").glob("*.yaml"))
        self.assertEqual(len(configs), 4)
        launcher_scripts = list(
            (self.build / "03-proposal/launcher-scripts").glob("*.sh")
        )
        self.assertEqual(len(launcher_scripts), 4)
        self.assertTrue(all(path.stat().st_mode & 0o100 for path in launcher_scripts))
        self.assertTrue(
            all(
                "python -m dolma.tokenizer.reshard" in path.read_text()
                for path in launcher_scripts
            )
        )
        self.assertTrue(
            all(
                "RESHARDING_MANIFEST_SCHEMA_VERSION" in path.read_text()
                for path in launcher_scripts
            )
        )
        for path in launcher_scripts:
            self.assertEqual(
                subprocess.run(["bash", "-n", path], check=False).returncode,
                0,
            )
        with (self.build / "03-proposal/config-index.csv").open() as f:
            execution_units = list(csv.DictReader(f))
        self.assertTrue(
            all(
                int(row["estimated_peak_local_bytes"])
                <= int(row["max_unit_working_bytes"])
                for row in execution_units
            )
        )
        dataset_layout = json.loads(
            (self.build / "03-proposal/dataset-layout.json").read_text()
        )
        self.assertEqual(dataset_layout["category_count"], 3)
        self.assertEqual(dataset_layout["execution_unit_count"], 4)
        runtime_requirements = json.loads(
            (self.build / "03-proposal/runtime-requirements.json").read_text()
        )
        self.assertEqual(
            runtime_requirements["required_resharding_manifest_schema_version"], 1
        )
        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            self.assertFalse(config["allow_existing_destination"])
            self.assertTrue(config["source_manifests"])
            parsed = ReshardingConfig.from_file(config_path)
            self.assertEqual(len(parsed.source_manifests), 1)
            self.assertTrue(Path(parsed.source_manifests[0].manifest).is_file())
            self.assertIn(
                "/new-datasets/dolma3p5/dolma3p5-14t-", config["destination_prefix"]
            )
        for plot in (self.build / "03-proposal/plots").glob("*.svg"):
            ElementTree.parse(plot)

        with (self.build / "02-inventory/normalized-s3-inventory.csv").open() as f:
            inventory_rows = [
                row for row in csv.DictReader(f) if row["required"] == "true"
            ]
        source_objects = [
            S3Object(
                bucket=row["bucket"],
                key=row["key"],
                size_bytes=int(row["size_bytes"]),
                etag=row["etag"],
                last_modified=row["last_modified"],
            )
            for row in inventory_rows
        ]
        client = MagicMock()
        client.list_objects_v2.return_value = {}
        session = MagicMock()
        session.client.return_value = client
        with (
            patch(
                "scripts.dolma3p5_resharding.workflow.boto3.Session",
                return_value=session,
            ),
            patch(
                "scripts.dolma3p5_resharding.workflow._list_prefix",
                return_value=source_objects,
            ),
        ):
            preflight_build(
                argparse.Namespace(
                    build=self.build,
                    profile=None,
                    region=None,
                    max_workers=2,
                )
            )

        with (self.build / "03-proposal/config-index.csv").open() as f:
            config_index = list(csv.DictReader(f))
        output_by_prefix = {}
        for row in config_index:
            destination = row["destination_prefix"].removeprefix("s3://")
            bucket, prefix = destination.split("/", 1)
            prefix = prefix.rstrip("/") + "/"
            planned_bytes = int(row["planned_uint32_values"]) * 4
            output_by_prefix[(bucket, prefix)] = [
                S3Object(bucket, prefix + "000000.npy", planned_bytes, "output-etag"),
                S3Object(bucket, prefix + "000000.csv.gz", 24, "metadata-etag"),
            ]

        def output_listing(_client, bucket, prefix):
            return output_by_prefix[(bucket, prefix)]

        with (
            patch(
                "scripts.dolma3p5_resharding.workflow.boto3.Session",
                return_value=session,
            ),
            patch(
                "scripts.dolma3p5_resharding.workflow._list_prefix",
                side_effect=output_listing,
            ),
        ):
            verify_output(
                argparse.Namespace(
                    build=self.build,
                    profile=None,
                    region=None,
                    max_workers=2,
                )
            )
        validate_build(argparse.Namespace(build=self.build))
        with self.assertRaises(PreparationError):
            propose_configs(
                argparse.Namespace(
                    build=self.build,
                    destination_root="s3://test-bucket/new-datasets/dolma3p5",
                    local_temp_root=str(self.root / "temp-base"),
                    max_unit_working_bytes=20_000_000_000_000,
                )
            )

    def test_allocation_is_deterministic_and_size_based(self):
        first = _allocate_object_repetitions(26, [8, 12, 20])
        second = _allocate_object_repetitions(26, [8, 12, 20])
        self.assertEqual(first, second)
        repetitions, planned = first
        self.assertEqual(
            planned,
            sum(size * repeat for size, repeat in zip([8, 12, 20], repetitions)),
        )
        self.assertLessEqual(abs(planned - 26), 6)

    def test_category_execution_units_respect_working_budget(self):
        row = {
            "npy_uri": "s3://bucket/tokens.npy",
            "metadata_uri": "s3://bucket/tokens.csv.gz",
            "npy_size_bytes": 400,
            "metadata_size_bytes": 20,
            "repeat_count": 7,
        }
        units = _partition_object_uses([row], max_unit_working_bytes=1_700)
        self.assertEqual([unit[0]["repeat_count"] for unit in units], [3, 3, 1])
        self.assertEqual(sum(unit[0]["repeat_count"] for unit in units), 7)
        with self.assertRaises(PreparationError):
            _partition_object_uses([row], max_unit_working_bytes=839)


class TestReshardingSafety(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _local_manifest(self) -> Path:
        pair = self.root / "pair"
        pair.mkdir()
        npy = pair / "tokens.npy"
        metadata = pair / "tokens.csv.gz"
        npy.write_bytes(b"\x00" * 16)
        metadata.write_text("0,1,id,src,0\n")
        manifest = self.root / "manifest.csv"
        with manifest.open("x", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["npy_uri", "metadata_uri", "repeat_count"]
            )
            writer.writeheader()
            writer.writerow(
                {"npy_uri": npy, "metadata_uri": metadata, "repeat_count": 2}
            )
        return manifest

    def test_manifest_preserves_exact_local_pair_and_repetition(self):
        manifest = ReshardingManifestConfig(self._local_manifest())
        paths = manifest.take(self.root / "run-input", max_workers=1)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], paths[1])
        self.assertTrue(Path(paths[0].npy_path).is_file())

    def test_remote_manifest_refuses_source_drift_before_download(self):
        manifest_path = self.root / "remote-manifest.csv"
        with manifest_path.open("x", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "npy_uri",
                    "metadata_uri",
                    "repeat_count",
                    "npy_size_bytes",
                    "metadata_size_bytes",
                    "npy_etag",
                    "metadata_etag",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "npy_uri": "s3://source-bucket/path/tokens.npy",
                    "metadata_uri": "s3://source-bucket/path/tokens.csv.gz",
                    "repeat_count": 1,
                    "npy_size_bytes": 16,
                    "metadata_size_bytes": 12,
                    "npy_etag": "approved-npy",
                    "metadata_etag": "approved-metadata",
                }
            )
        client = MagicMock()
        client.head_object.return_value = {
            "ContentLength": 20,
            "ETag": '"changed"',
        }
        with patch("dolma.tokenizer.reshard.boto3.client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "changed before download"):
                ReshardingManifestConfig(manifest_path).take(
                    self.root / "remote-input", max_workers=1
                )

    def test_existing_local_destination_is_refused(self):
        destination = self.root / "existing"
        destination.mkdir()
        self.assertTrue(destination_has_objects(destination))
        config = ReshardingConfig.from_dict(
            {
                "destination_prefix": str(destination),
                "source_manifests": [{"manifest": str(self._local_manifest())}],
                "max_num_files": 2,
            }
        )
        with self.assertRaises(FileExistsError):
            reshard(config)

    def test_cleanup_only_removes_run_owned_child(self):
        manifest = self._local_manifest()
        temp_base = self.root / "temp-base"
        temp_base.mkdir()
        sentinel = temp_base / "keep-me"
        sentinel.write_text("sentinel")
        destination = self.root / "new-output"
        config = ReshardingConfig.from_dict(
            {
                "destination_prefix": str(destination),
                "source_manifests": [{"manifest": str(manifest)}],
                "local_tempdir": str(temp_base),
                "max_num_files": 2,
            }
        )
        with patch("dolma.tokenizer.reshard.merge_all_npys"):
            reshard(config)
        self.assertEqual(sentinel.read_text(), "sentinel")
        self.assertEqual([path.name for path in temp_base.iterdir()], ["keep-me"])

    def test_upload_always_uses_no_clobber(self):
        completed = MagicMock(returncode=0, stdout="", stderr="")
        with patch(
            "dolma.tokenizer.reshard.subprocess.run", return_value=completed
        ) as run:
            upload_to_s3(self.root, "s3://test-bucket/new/prefix", max_workers=3)
        command = run.call_args.args[0]
        self.assertIn("--no-clobber", command)
        self.assertEqual(command[:3], ["s5cmd", "--numworkers", "3"])

    def test_s3_bucket_root_is_refused_before_listing(self):
        with self.assertRaises(ValueError):
            destination_has_objects("s3://test-bucket")
        client = MagicMock()
        client.list_objects_v2.return_value = {"Contents": [{"Key": "existing"}]}
        with patch("dolma.tokenizer.reshard.boto3.client", return_value=client):
            self.assertTrue(destination_has_objects("s3://test-bucket/new/prefix"))


if __name__ == "__main__":
    unittest.main()
