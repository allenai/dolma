import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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
    _finalize_inventory,
    _load_catalog,
    _parse_s5cmd_jsonl,
    _partition_object_uses,
    collect_inventory,
    plan_build,
    preflight_build,
    propose_configs,
    refresh_inventory_details,
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
                        },
                        {
                            "name": "dropped",
                            "weight": 0.0,
                            "paths": [
                                "dolma3p5_pool/catalog-source/topic/vigintile_0000/allenai/tokenizer/*.npy"
                            ],
                            "repetition_factor": -1.0,
                        },
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
            "ai2-llm,preprocessed/catalog-source/topic/vigintile_0000/allenai/tokenizer/0000.npy\n"
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
        phase.mkdir()
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
            "s3://ai2-llm/preprocessed/catalog-source/topic/vigintile_0000/allenai/tokenizer/0000.npy": 200,
            "s3://ai2-llm/preprocessed/catalog-source/topic/vigintile_0000/allenai/tokenizer/0000.csv.gz": 16,
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

    def test_inventory_uses_s5cmd(self):
        self._plan()
        captured_command = []

        def run_collector(command, raw_output, environment, status):
            captured_command.extend(command)
            self.assertIn("AWS_PROFILE", environment)
            with raw_output.open("x") as stdout:
                for uri, size in self._inventory_objects().items():
                    stdout.write(
                        json.dumps(self._inventory_record(uri, size)) + "\n"
                    )
            status("Running: 8 JSON records received, 1s elapsed")
            return SimpleNamespace(
                returncode=0,
                stderr="",
                output_records=len(self._inventory_objects()),
                elapsed_seconds=1.0,
            )

        session = MagicMock()
        output = io.StringIO()
        with redirect_stdout(output):
            with (
                patch(
                    "scripts.dolma3p5_resharding.workflow.shutil.which",
                    return_value="/usr/bin/s5cmd",
                ),
                patch(
                    "scripts.dolma3p5_resharding.workflow._run_s5cmd_inventory",
                    side_effect=run_collector,
                ),
                patch(
                    "scripts.dolma3p5_resharding.workflow.boto3.Session",
                    return_value=session,
                ),
            ):
                inventory_args = argparse.Namespace(
                    build=self.build,
                    profile="read-only",
                    region="us-west-2",
                    max_workers=2,
                )
                collect_inventory(inventory_args)
                collect_inventory(inventory_args)

        self.assertEqual(captured_command[0], "s5cmd")
        self.assertEqual(captured_command.count("s5cmd"), 2)
        collector = json.loads((self.build / "02-inventory/collector.json").read_text())
        self.assertEqual(collector["collector"], "s5cmd")
        self.assertEqual(collector["output_records"], 8)
        self.assertTrue((self.build / "02-inventory/raw-listings.jsonl").is_file())
        status_output = output.getvalue()
        self.assertIn("[inventory 1/4] Bulk listing:", status_output)
        self.assertIn("[inventory 1/4] Running:", status_output)
        self.assertIn("[inventory 2/4] Resolving required objects:", status_output)
        self.assertIn("[inventory 3/4] Validation:", status_output)
        self.assertIn("[inventory 3/4] Estimated source tokens:", status_output)
        self.assertIn("[inventory 4/4] PASS:", status_output)

    def test_inventory_requires_s5cmd_without_replacing_existing_artifacts(self):
        self._plan()
        inventory_phase = self.build / "02-inventory"
        inventory_phase.mkdir()
        existing = inventory_phase / "keep.txt"
        existing.write_text("existing inventory")

        with (
            patch(
                "scripts.dolma3p5_resharding.workflow.shutil.which",
                return_value=None,
            ),
            self.assertRaisesRegex(PreparationError, "s5cmd is required"),
        ):
            collect_inventory(
                argparse.Namespace(
                    build=self.build,
                    profile=None,
                    region=None,
                    max_workers=None,
                )
            )

        self.assertEqual(existing.read_text(), "existing inventory")

    def test_plan_reports_resolution_counts_without_chart_artifacts(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self._plan()
        self.assertIn(
            "Plan summary: 3 catalog NPY matches, 1 direct S3 pattern, "
            "1 correction, 0 blocking failures",
            output.getvalue(),
        )
        self.assertFalse(
            (self.build / "01-plan/plots/resolution-counts.svg").exists()
        )
        self.assertFalse(
            (self.build / "01-plan/plot-data/resolution-counts.csv").exists()
        )

    def test_plan_command_builds_paths_and_inventory_together(self):
        from scripts.dolma3p5_resharding import plan as plan_command

        output = self.root / "combined-plan"
        with (
            patch.object(plan_command.shutil, "which", return_value="/usr/bin/s5cmd"),
            patch.object(plan_command, "plan_build") as build_paths,
            patch.object(plan_command, "collect_inventory") as inventory,
            patch.object(
                sys,
                "argv",
                [
                    "plan.py",
                    "--mix",
                    str(self.mix),
                    "--catalog",
                    str(self.catalog),
                    "--output",
                    str(output),
                    "--profile",
                    "read-only",
                ],
            ),
        ):
            plan_command.main()

        build_paths.assert_called_once()
        inventory.assert_called_once()
        inventory_args = inventory.call_args.args[0]
        self.assertEqual(inventory_args.build, output)
        self.assertEqual(inventory_args.profile, "read-only")

    def test_catalog_paths_are_decoded_to_literal_s3_keys(self):
        catalog = self.root / "encoded-catalog.csv"
        catalog.write_text(
            "ai2-llm,preprocessed/the-stack-v2/C%2B%2B/0000.npy\n"
        )

        rows = _load_catalog(catalog)

        self.assertEqual(
            rows[0]["key"], "preprocessed/the-stack-v2/C++/0000.npy"
        )

    def test_s5cmd_listing_commands_use_literal_s3_keys(self):
        mix = self.root / "encoded-mix.yaml"
        catalog = self.root / "encoded-command-catalog.csv"
        build = self.root / "encoded-command-build"
        yaml_path = (
            "dolma3p5_pool/the-stack-v2/C++/quality_p95/"
            "allenai/dolma2-tokenizer/*.npy"
        )
        mix.write_text(
            yaml.safe_dump(
                {
                    "mix": [
                        {
                            "name": "the-stack-v2:C++",
                            "weight": 1.0,
                            "categories": [
                                {
                                    "name": "high",
                                    "weight": 1.0,
                                    "paths": [yaml_path],
                                    "repetition_factor": -1.0,
                                }
                            ],
                        }
                    ]
                },
                sort_keys=False,
            )
        )
        catalog.write_text(
            "ai2-llm,preprocessed/the-stack-v2/C%2B%2B/quality_p95/"
            "allenai/dolma2-tokenizer/0000.npy\n"
        )

        plan_build(
            argparse.Namespace(
                mix=mix,
                catalog=catalog,
                settings=None,
                output=build,
            )
        )

        commands = (build / "01-plan/bulk-listing-commands.txt").read_text()
        self.assertIn("/C++/quality_p95/", commands)
        self.assertNotIn("%2B", commands)

    def test_end_to_end_preparation_phases_are_replaceable(self):
        mix_before = self.mix.read_bytes()
        catalog_before = self.catalog.read_bytes()
        self._plan()
        plan_report = (self.build / "01-plan/report.html").read_text()
        self.assertIn(
            "Dolma 3.5 Target Allocation and Source Path Plan", plan_report
        )
        self.assertIn("Materialized output target:", plan_report)
        self.assertNotIn("S3 source volume:", plan_report)
        self.assertIn('data-detail="plan-family-detail-', plan_report)
        self.assertIn('class="subcategory-row', plan_report)
        self.assertIn('class="subcategory-detail"', plan_report)
        self.assertIn('class="category-grid"', plan_report)
        self.assertIn("matched NPY", plan_report)
        self.assertIn("topic", plan_report)
        with (self.build / "01-plan/normalized-paths.csv").open() as f:
            self.assertNotIn("resolution_route", csv.DictReader(f).fieldnames)
        with (self.build / "01-plan/listing-plan.csv").open() as f:
            self.assertNotIn("resolution_routes", csv.DictReader(f).fieldnames)
        plan_target_plot = (self.build / "01-plan/plots/target-mix.svg").read_text()
        self.assertIn(
            "Total target: 14T tokens (14,000,000,000,000)", plan_target_plot
        )
        self.assertIn("50.00% · 7T tokens", plan_target_plot)
        self.assertNotIn('text-anchor="end"', plan_target_plot)
        with (self.build / "01-plan/corrections.csv").open() as f:
            corrections = list(csv.DictReader(f))
        self.assertEqual(len(corrections), 1)
        self.assertIn("quality_p95", corrections[0]["path"])
        with (self.build / "01-plan/direct-s3-patterns.csv").open() as f:
            direct = list(csv.DictReader(f))
        self.assertEqual(len(direct), 1)
        finder_metadata = self.build / ".DS_Store"
        finder_metadata.write_bytes(b"preserve benign metadata")
        self._plan()
        self.assertEqual(self.mix.read_bytes(), mix_before)
        self.assertEqual(self.catalog.read_bytes(), catalog_before)
        self.assertEqual(finder_metadata.read_bytes(), b"preserve benign metadata")

        self._write_inventory()
        inventory_summary = json.loads(
            (self.build / "02-inventory/inventory-summary.json").read_text()
        )
        self.assertEqual(inventory_summary["source_count"], 3)
        self.assertEqual(inventory_summary["source_family_count"], 3)
        self.assertEqual(inventory_summary["subcategory_count"], 3)
        self.assertEqual(inventory_summary["category_count"], 4)
        self.assertEqual(inventory_summary["lower_group_count"], 4)
        self.assertEqual(inventory_summary["source_uint32_values"], 650)
        self.assertEqual(inventory_summary["token_delta"], 13_999_999_999_350)
        self.assertGreater(inventory_summary["sampling_ratio"], 1)
        self.assertIn("upsample", inventory_summary["sampling_rate"])
        self.assertEqual(
            inventory_summary["details_artifact"], "inventory-details.json"
        )
        inventory_details = json.loads(
            (self.build / "02-inventory/inventory-details.json").read_text()
        )
        self.assertEqual(inventory_details["source_uint32_values"], 650)
        self.assertEqual(inventory_details["source_family_count"], 3)
        self.assertEqual(inventory_details["subcategory_count"], 3)
        catalog_source = next(
            source
            for source in inventory_details["sources"]
            if source["mix_name"] == "catalog-source:topic"
        )
        self.assertEqual(catalog_source["source_uint32_values"], 150)
        self.assertEqual(catalog_source["source_family"], "catalog-source")
        self.assertEqual(catalog_source["subcategory_name"], "topic")
        dropped_category = next(
            category
            for category in catalog_source["categories"]
            if category["category_name"] == "dropped"
        )
        self.assertFalse(dropped_category["active"])
        self.assertEqual(dropped_category["source_uint32_values"], 50)
        self.assertAlmostEqual(
            dropped_category["source_percent_of_parent"], 100 / 3
        )
        self.assertEqual(
            dropped_category["lower_groups"][0]["lower_group"],
            "vigintile_0000",
        )
        self.assertEqual(
            dropped_category["lower_groups"][0]["source_percent_of_parent"],
            100,
        )
        refreshed_summary = refresh_inventory_details(self.build)
        self.assertEqual(refreshed_summary["source_count"], 3)
        inventory_plot = (
            self.build / "02-inventory/plots/available-by-mix.svg"
        ).read_text()
        self.assertIn("% ·", inventory_plot)
        self.assertIn("tokens", inventory_plot)
        inventory_report = (self.build / "02-inventory/report.html").read_text()
        self.assertIn('data-detail="inventory-family-detail-', inventory_report)
        self.assertIn('class="subcategory-row', inventory_report)
        self.assertIn('class="subcategory-detail"', inventory_report)
        self.assertIn("const setAccordionState", inventory_report)
        self.assertIn(
            "button.getAttribute('aria-expanded') !== 'true'", inventory_report
        )
        self.assertIn('class="comparison-bars"', inventory_report)
        self.assertIn('class="category-metrics"', inventory_report)
        self.assertNotIn('class="category-counts"', inventory_report)
        self.assertIn('class="path-stat"', inventory_report)
        self.assertIn('class="path-metric"', inventory_report)
        self.assertIn('class="mix-metric-label">Sampling', inventory_report)
        self.assertIn("Source Inventory and Sampling Plan", inventory_report)
        self.assertIn('class="summary-metrics"', inventory_report)
        self.assertIn('class="mix-metrics"', inventory_report)
        self.assertIn("Overall sampling", inventory_report)
        self.assertNotIn("S3", inventory_report)
        self.assertNotIn("NPY bytes", inventory_report)
        self.assertNotIn("(+", inventory_report)
        self.assertNotIn("(−", inventory_report)
        self.assertIn("source ", inventory_report)
        self.assertIn("% of entry", inventory_report)
        self.assertIn("% of category", inventory_report)
        self.assertIn("upsample", inventory_report)
        self.assertIn("downsample", inventory_report)
        self.assertIn("vigintile_0000", inventory_report)
        sampling_path = (
            self.build / "02-inventory/plot-data/sampling-by-lower-group.csv"
        )
        with sampling_path.open() as f:
            sampling_paths = list(csv.DictReader(f))
        dropped = next(row for row in sampling_paths if row["lower_group"] == "vigintile_0000")
        self.assertEqual(dropped["original_uint32_values"], "50")
        self.assertEqual(dropped["implied_target_uint32_values"], "0")
        self.assertIn("downsample", dropped["sampling_rate"])
        with (self.build / "02-inventory/required-objects.csv").open() as f:
            self.assertNotIn("resolution_route", csv.DictReader(f).fieldnames)
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
        proposal_target_plot = (
            self.build / "03-proposal/plots/target-mix.svg"
        ).read_text()
        self.assertIn("50.00% · 7T tokens", proposal_target_plot)
        proposal_report = (self.build / "03-proposal/report.html").read_text()
        self.assertIn("Pre-materialization Sampling Proposal", proposal_report)
        self.assertIn('data-detail="proposal-family-detail-', proposal_report)
        self.assertIn('class="subcategory-row', proposal_report)
        self.assertIn('class="subcategory-detail"', proposal_report)
        self.assertIn('class="summary-metrics"', proposal_report)
        self.assertIn('class="mix-metrics"', proposal_report)
        self.assertIn("Overall sampling", proposal_report)
        self.assertNotIn("S3", proposal_report)
        self.assertNotIn("NPY bytes", proposal_report)
        self.assertNotIn("(+", proposal_report)
        self.assertNotIn("(−", proposal_report)
        self.assertIn("source →", proposal_report)
        self.assertIn("proposed", proposal_report)
        self.assertIn("× upsample", proposal_report)
        self.assertIn('class="category-metrics proposal-metrics"', proposal_report)
        self.assertIn('class="path-metric"', proposal_report)
        self.assertIn('class="path-use-summary"', proposal_report)
        self.assertIn("per-object repeats", proposal_report)
        self.assertIn("vigintile_0000", proposal_report)
        with (self.build / "03-proposal/category-allocation.csv").open() as f:
            allocation_rows = list(csv.DictReader(f))
        catalog_allocation = next(
            row
            for row in allocation_rows
            if row["mix_name"] == "catalog-source:topic"
        )
        self.assertEqual(catalog_allocation["available_uint32_values"], "100")
        self.assertEqual(
            int(catalog_allocation["planned_uint32_values"])
            - int(catalog_allocation["available_uint32_values"]),
            int(catalog_allocation["token_change_from_original"]),
        )
        self.assertEqual(
            catalog_allocation["minimum_repetition"],
            catalog_allocation["maximum_repetition"],
        )
        self.assertGreater(int(catalog_allocation["maximum_repetition"]), 1)
        with (
            self.build
            / "03-proposal/plot-data/proposed-sampling-by-lower-group.csv"
        ).open() as f:
            proposed_paths = list(csv.DictReader(f))
        dropped_proposal = next(
            row for row in proposed_paths if row["lower_group"] == "vigintile_0000"
        )
        self.assertEqual(dropped_proposal["original_uint32_values"], "50")
        self.assertEqual(dropped_proposal["proposed_uint32_values"], "0")
        self.assertEqual(dropped_proposal["token_change_from_original"], "-50")
        self.assertEqual(dropped_proposal["minimum_repetition"], "0")
        self.assertEqual(dropped_proposal["maximum_repetition"], "0")
        self.assertEqual(dropped_proposal["dropped_object_count"], "1")
        self.assertEqual(dropped_proposal["repeated_object_count"], "0")
        proposal_summary = json.loads(
            (self.build / "03-proposal/proposal-summary.json").read_text()
        )
        self.assertEqual(proposal_summary["source_uint32_values"], 650)
        self.assertEqual(
            proposal_summary["token_change_from_source"],
            proposal_summary["planned_uint32_values"] - 650,
        )

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
        propose_configs(
            argparse.Namespace(
                build=self.build,
                destination_root="s3://test-bucket/new-datasets/dolma3p5",
                local_temp_root=str(self.root / "temp-base"),
                max_unit_working_bytes=20_000_000_000_000,
            )
        )
        self.assertTrue((self.build / "03-proposal/proposal-summary.json").is_file())
        self.assertFalse((self.build / "04-preflight").exists())
        self.assertFalse((self.build / "05-output-validation").exists())

    def test_plan_refuses_to_replace_unknown_files_in_a_build(self):
        self._plan()
        critical_file = self.build / "source-tokens.npy"
        critical_file.write_bytes(b"do not replace")

        with self.assertRaisesRegex(PreparationError, "unknown top-level entries"):
            self._plan()

        self.assertEqual(critical_file.read_bytes(), b"do not replace")

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
