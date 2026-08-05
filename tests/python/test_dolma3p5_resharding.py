import argparse
import csv
import gzip
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from xml.etree import ElementTree

import numpy as np
import yaml
from rich.console import Console

WORKER_STORAGE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/dolma3p5_resharding/setup_worker_storage.sh"
)

from dolma.tokenizer.reshard import (
    ReshardingConfig,
    ReshardingManifestConfig,
    destination_has_objects,
    merge_group,
    reshard,
    upload_to_s3,
)
from scripts.dolma3p5_resharding.materialize import (
    ClusterInstance,
    MaterializationGroup,
    PlannedWorkerGroup,
    WorkerAssignment,
    _execute_materialization_groups,
    _map_command,
    _partition_worker_rows,
    _planned_worker_groups,
    _prepare_workers,
    _print_dispatch,
    _provision_batches,
    _retag_cluster_instances,
    _run_compact_process,
    _run_selected_preflight,
    _safe_path_launcher_payload,
    _select_units,
    _status_detail,
    _verify_materialized_units,
    _wait_command,
    _wait_for_workers_to_stop,
    _worker_counts_for_groups,
    _worker_log_command,
    _worker_log_message,
    _worker_log_parts,
    _worker_log_snapshots,
)
from scripts.dolma3p5_resharding.materialize import (
    build_parser as build_materialize_parser,
)
from scripts.dolma3p5_resharding.workflow import (
    DEFAULT_REGION,
    EXECUTION_UNIT_INDEX_WIDTH,
    PreparationError,
    S3Object,
    _allocate_object_sampling,
    _category_output_directory,
    _execution_unit_work,
    _filter_execution_units,
    _finalize_inventory,
    _load_catalog,
    _parse_s5cmd_jsonl,
    _partition_object_uses,
    _planned_output_shards,
    _select_worker_instance,
    _self_contained_launcher,
    _source_relative_directory,
    _unit_selection_digest,
    _validate_execution_layout,
    collect_inventory,
    normalize_region,
    plan_build,
    preflight_build,
    propose_configs,
    refresh_inventory_details,
    validate_build,
    verify_output,
)


class TestDolma35ReshardingPreparation(unittest.TestCase):
    def test_output_shard_planning_is_size_based_and_bounded(self):
        target = 64 * 1024**3
        self.assertEqual(
            _planned_output_shards(
                output_npy_bytes=target // 2,
                input_view_count=20,
                target_output_shard_bytes=target,
                max_output_shards_per_unit=8,
            ),
            1,
        )
        self.assertEqual(
            _planned_output_shards(
                output_npy_bytes=target * 3,
                input_view_count=20,
                target_output_shard_bytes=target,
                max_output_shards_per_unit=8,
            ),
            3,
        )
        self.assertEqual(
            _planned_output_shards(
                output_npy_bytes=target * 20,
                input_view_count=20,
                target_output_shard_bytes=target,
                max_output_shards_per_unit=8,
            ),
            8,
        )
        self.assertEqual(
            _planned_output_shards(
                output_npy_bytes=target * 3,
                input_view_count=2,
                target_output_shard_bytes=target,
                max_output_shards_per_unit=8,
            ),
            2,
        )

    def test_execution_work_and_worker_grid_account_for_cpu_and_disk(self):
        rows = [
            {
                "estimated_uint32_values": 100,
                "repeat_count": 1,
                "partial_target_uint32_values": 30,
            }
        ]
        work = _execution_unit_work(rows, document_selection_work_passes=2)
        self.assertEqual(work["planned_uint32_values"], 130)
        self.assertEqual(work["document_selection_source_uint32_values"], 100)
        self.assertEqual(work["estimated_work_uint32_values"], 330)

        grid = [
            {
                "instance_type": "i4i.2xlarge",
                "vcpus": 8,
                "local_nvme_devices": 1,
                "local_nvme_bytes": 1_875,
                "max_estimated_work_uint32_values": 150,
            },
            {
                "instance_type": "i4i.4xlarge",
                "vcpus": 16,
                "local_nvme_devices": 1,
                "local_nvme_bytes": 3_750,
                "max_estimated_work_uint32_values": 300,
            },
            {
                "instance_type": "i4i.8xlarge",
                "vcpus": 32,
                "local_nvme_devices": 2,
                "local_nvme_bytes": 7_500,
                "max_estimated_work_uint32_values": None,
            },
        ]
        self.assertEqual(
            _select_worker_instance(
                grid,
                estimated_peak_local_bytes=1_000,
                estimated_work_uint32_values=100,
                disk_headroom_ratio=1.1,
            )["instance_type"],
            "i4i.2xlarge",
        )
        self.assertEqual(
            _select_worker_instance(
                grid,
                estimated_peak_local_bytes=2_000,
                estimated_work_uint32_values=100,
                disk_headroom_ratio=1.1,
            )["instance_type"],
            "i4i.4xlarge",
        )
        largest = _select_worker_instance(
            grid,
            estimated_peak_local_bytes=1_000,
            estimated_work_uint32_values=330,
            disk_headroom_ratio=1.1,
        )
        self.assertEqual(largest["instance_type"], "i4i.8xlarge")
        self.assertEqual(largest["storage_layout"], "raid0")

    def test_materialize_dry_run_uses_terse_cli_output(self):
        output = io.StringIO()
        with redirect_stdout(output):
            _print_dispatch(
                "category-example",
                [
                    {
                        "leaf_id": "000:example",
                        "unit_id": "00000000",
                        "planned_uint32_values": "1000000000",
                        "planned_output_shard_count": "1",
                        "estimated_peak_local_bytes": "2000000000",
                        "destination_prefix": "s3://bucket/output/00000000",
                    }
                ],
                Path("/tmp/launchers"),
                [("create missing workers", ["pmr", "create", "--number", "1"])],
                1,
                False,
            )
        rendered = output.getvalue()
        self.assertIn("Dry run", rendered)
        self.assertIn("Selection", rendered)
        self.assertIn("example", rendered)
        self.assertIn("1 category · 1 unit · 1 worker", rendered)
        self.assertIn("commands:\npmr create --number 1", rendered)
        self.assertNotIn("Worker lifecycle", rendered)
        self.assertNotIn("create missing workers:", rendered)

    def test_materialize_execute_summary_omits_dry_run_details(self):
        output = io.StringIO()
        with redirect_stdout(output):
            _print_dispatch(
                "category-example",
                [
                    {
                        "leaf_id": "000:example",
                        "unit_id": "00000000",
                        "planned_uint32_values": "1000000000",
                        "planned_output_shard_count": "1",
                        "estimated_peak_local_bytes": "2000000000",
                        "destination_prefix": "s3://bucket/output/00000000",
                    }
                ],
                Path("/tmp/launchers"),
                [("create missing workers", ["pmr", "create", "--number", "1"])],
                1,
                True,
                cluster="dolma3p5-14t",
                project="oe-other",
                region="us-east-1",
            )
        rendered = output.getvalue()
        self.assertIn("Execution", rendered)
        self.assertIn("Selection", rendered)
        self.assertIn("1 category · 1 unit · 1 worker", rendered)
        self.assertIn("Cluster", rendered)
        self.assertIn("dolma3p5-14t", rendered)
        self.assertIn("Project", rendered)
        self.assertIn("oe-other", rendered)
        self.assertIn("Region", rendered)
        self.assertIn("us-east-1", rendered)
        self.assertNotIn("category-example", rendered)
        self.assertNotIn("commands:", rendered)
        self.assertNotIn("pmr create", rendered)
        self.assertNotIn("units:", rendered)

    def test_compact_process_discards_success_output(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, color_system=None)
        return_code = _run_compact_process(
            "wait for workers",
            [
                sys.executable,
                "-c",
                (
                    "print('Waiting for instances... 0/2 ready (0s)'); "
                    "print('  · worker-0000 [running]'); "
                    "print('Waiting for instances... 2/2 ready (10s)')"
                ),
            ],
            console=console,
        )
        rendered = output.getvalue()
        self.assertEqual(return_code, 0)
        self.assertIn("wait for workers", rendered)
        self.assertIn("✓", rendered)
        self.assertNotIn("Waiting for instances", rendered)
        self.assertNotIn("worker-0000", rendered)

    def test_compact_process_streams_child_output_when_verbose(self):
        output = io.StringIO()
        return_code = _run_compact_process(
            "prepare local NVMe",
            [sys.executable, "-c", "print('worker setup detail')"],
            console=Console(file=output, force_terminal=False, color_system=None),
            verbose=True,
        )
        self.assertEqual(return_code, 0)
        self.assertIn("worker setup detail", output.getvalue())

    def test_compact_process_bounds_and_redacts_failure_output(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, color_system=None)
        return_code = _run_compact_process(
            "create workers",
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "[print(f'line-{index}') for index in range(20)]; "
                    "print('aws_secret_access_key=do-not-print'); "
                    "sys.exit(3)"
                ),
            ],
            console=console,
        )
        rendered = output.getvalue()
        self.assertEqual(return_code, 3)
        self.assertIn("✗ create workers", rendered)
        self.assertIn("last output:", rendered)
        self.assertNotIn("line-0\n", rendered)
        self.assertIn("line-19", rendered)
        self.assertIn("aws_secret_access_key=<redacted>", rendered)
        self.assertNotIn("do-not-print", rendered)

    def test_compact_status_hides_poormanray_cluster_log_mislabeled_as_project(self):
        self.assertIsNone(
            _status_detail(
                "prepare local NVMe",
                "[INFO][21:31:31] Running command on instances with "
                "project=dolma3p5-14t in region us-east-1 (aws)",
            )
        )
        self.assertEqual(
            _status_detail(
                "submit materialization",
                "[INFO][21:31:36] Job 123 started on 2 instances.",
            ),
            "accepted by 2 workers",
        )

    def test_worker_log_message_removes_envelope_but_preserves_errors(self):
        self.assertEqual(
            _worker_log_parts(
                "[2026-08-05 15:38:11 main.dolma.__main__ INFO] merge 50.0%"
            ),
            ("2026-08-05 15:38:11", "merge 50.0%"),
        )
        self.assertEqual(
            _worker_log_message(
                "[2026-08-05 15:38:11 main.dolma.__main__ INFO] merge 50.0%"
            ),
            "merge 50.0%",
        )
        self.assertEqual(
            _worker_log_message(
                "[2026-08-05 15:38:11 main.dolma.__main__ ERROR] disk full"
            ),
            "error · disk full",
        )

    def test_materialize_waits_for_worker_spindown(self):
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            profile=None,
            completion_poll_seconds=30,
            verbose=True,
        )
        describe = MagicMock(
            side_effect=[
                [
                    ClusterInstance("i-first", "running", "i4i.2xlarge", "oe-other"),
                    ClusterInstance("i-second", "stopping", "i4i.2xlarge", "oe-other"),
                ],
                [
                    ClusterInstance("i-first", "running", "i4i.2xlarge", "oe-other"),
                    ClusterInstance("i-second", "running", "i4i.2xlarge", "oe-other"),
                ],
                [
                    ClusterInstance("i-first", "stopped", "i4i.2xlarge", "oe-other"),
                    ClusterInstance("i-second", "stopped", "i4i.2xlarge", "oe-other"),
                ],
            ]
        )
        sleep = MagicMock()
        output = io.StringIO()
        with patch(
            "scripts.dolma3p5_resharding.materialize._worker_log_snapshots",
            side_effect=[
                {
                    "i-first": {
                        "statuses": {"00000000": "running"},
                        "logs": {
                            "00000000.log": ("Downloading exact manifest objects",),
                        },
                    }
                },
                {
                    "i-first": {
                        "statuses": {"00000000": "running"},
                        "logs": {
                            "00000000.log": (
                                "Downloading exact manifest objects",
                                (
                                    "[2026-08-05 15:38:11 main.dolma.__main__ INFO] "
                                    "merge 50.0% · 10B/20B tokens · 40M tokens/s"
                                ),
                            ),
                        },
                    },
                    "i-second": {
                        "statuses": {"00000001": "running"},
                        "logs": {
                            "00000001.log": ("Other worker progress",),
                        },
                    },
                },
            ],
        ):
            _wait_for_workers_to_stop(
                args,
                ["i-first", "i-second"],
                2,
                "test-run",
                describe=describe,
                sleep=sleep,
                console=Console(file=output, force_terminal=False, color_system=None),
            )
        self.assertEqual(sleep.call_args_list, [call(30), call(30)])
        rendered = output.getvalue()
        self.assertIn("2 units · 1 running · 1 stopping · 0 stopped", rendered)
        self.assertIn("] [-first] 00000000 · running", rendered)
        self.assertIn("] [-first] unit 00000000", rendered)
        self.assertEqual(rendered.count("unit 00000000"), 1)
        self.assertIn("Downloading exact manifest objects", rendered)
        self.assertEqual(rendered.count("Downloading exact manifest objects"), 1)
        self.assertIn("merge 50.0% · 10B/20B tokens · 40M tokens/s", rendered)
        self.assertNotIn("main.dolma.__main__ INFO", rendered)
        self.assertIn("] [second] Other worker progress", rendered)
        self.assertIn(
            "[2026-08-05 15:38:11] [-first] merge 50.0%", rendered
        )
        self.assertNotIn("[i-first]", rendered)
        self.assertNotIn("[i-second]", rendered)
        self.assertNotIn("worker 01", rendered)
        self.assertIn("materialization workers stopped", rendered)

    def test_materialize_status_tracks_workers_in_different_lifecycle_stages(self):
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            profile=None,
            completion_poll_seconds=30,
            verbose=False,
        )
        describe = MagicMock(
            side_effect=[
                [
                    ClusterInstance("i-ready", "running", "i4i.2xlarge", "oe-other"),
                    ClusterInstance("i-slow", "running", "i4i.2xlarge", "oe-other"),
                ],
                [
                    ClusterInstance("i-ready", "stopped", "i4i.2xlarge", "oe-other"),
                    ClusterInstance("i-slow", "running", "i4i.2xlarge", "oe-other"),
                ],
                [
                    ClusterInstance("i-ready", "stopped", "i4i.2xlarge", "oe-other"),
                    ClusterInstance("i-slow", "stopped", "i4i.2xlarge", "oe-other"),
                ],
            ]
        )
        worker_stages = {"i-ready": "materializing", "i-slow": "waiting"}
        stage_lock = Lock()
        all_dispatched = Event()

        sleep_calls = 0

        def advance_slow_worker(_seconds):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 1:
                with stage_lock:
                    worker_stages["i-slow"] = "materializing"
                all_dispatched.set()

        sleep = MagicMock(side_effect=advance_slow_worker)
        output = io.StringIO()

        _wait_for_workers_to_stop(
            args,
            ["i-ready", "i-slow"],
            2,
            "test-run",
            describe=describe,
            sleep=sleep,
            console=Console(
                file=output, force_terminal=False, color_system=None, width=300
            ),
            worker_stages=worker_stages,
            stage_lock=stage_lock,
            all_dispatched=all_dispatched,
        )

        rendered = output.getvalue()
        self.assertIn("1 waiting", rendered)
        self.assertIn("1 materializing", rendered)
        self.assertIn("1 stopped", rendered)
        self.assertIn("2 stopped", rendered)
        self.assertIn("materialization workers stopped", rendered)

    @patch("scripts.dolma3p5_resharding.materialize.subprocess.run")
    def test_verbose_worker_logs_are_complete_for_every_worker(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=(
                "Instance i-first:\n"
                "stdout: @@DOLMA_STATUS@@\t00000000\trunning\n"
                "@@DOLMA_LOG_BEGIN@@\t00000000.log\n"
                "first line\n"
                "12.00%  12 GB / 100 GB (1.20 GB/s) 1m left (0/1)\r"
                "24.00%  24 GB / 100 GB (1.18 GB/s) 1m left (0/1)\r"
                "second line\n"
                "@@DOLMA_LOG_END@@\t00000000.log\n"
                "stderr:\n\n"
                "Instance i-second:\n"
                "stdout: @@DOLMA_STATUS@@\t00000001\tsucceeded\n"
                "@@DOLMA_LOG_BEGIN@@\t00000001.log\n"
                "other worker line\n"
                "@@DOLMA_LOG_END@@\t00000001.log\n"
                "stderr:\n"
            ),
            stderr="",
        )
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=2,
            ssh_key_path=None,
        )
        snapshots = _worker_log_snapshots(
            args,
            ["i-first", "i-second"],
            "test-run",
        )
        self.assertEqual(
            snapshots["i-first"]["logs"]["00000000.log"],
            (
                "first line",
                "12.00%  12 GB / 100 GB (1.20 GB/s) 1m left (0/1)",
                "24.00%  24 GB / 100 GB (1.18 GB/s) 1m left (0/1)",
                "second line",
            ),
        )
        self.assertEqual(
            snapshots["i-second"]["logs"]["00000001.log"],
            ("other worker line",),
        )

    def test_materialize_verifies_selected_output_sizes_and_metadata(self):
        args = SimpleNamespace(
            profile=None,
            region="us-east-1",
            parallelism=2,
        )
        row = {
            "unit_id": "00000000",
            "destination_prefix": "s3://bucket/output/00000000",
            "planned_uint32_values": "10",
            "planned_output_shard_count": "1",
            "allowed_materialized_target_residual_uint32_values": "0",
        }
        objects = [
            S3Object("bucket", "output/00000000/000000.npy", 40),
            S3Object("bucket", "output/00000000/000000.csv.gz", 20),
        ]
        output = io.StringIO()
        with patch(
            "scripts.dolma3p5_resharding.materialize._list_prefix",
            return_value=objects,
        ):
            checks = _verify_materialized_units(
                args,
                [row],
                client=MagicMock(),
                console=Console(file=output, force_terminal=False, color_system=None),
            )
        self.assertEqual(checks[0].actual_uint32_values, 10)
        self.assertEqual(checks[0].metadata_count, 1)
        self.assertIn("materialization verified  1/1 units", output.getvalue())

    def test_generated_materialize_launcher_uses_safe_import_directory(self):
        launcher = _self_contained_launcher(
            unit_id="00000000",
            config_name="00000000.yaml",
            config_text="destination_prefix: /tmp/output\n",
            manifest_name="00000000.csv",
            manifest_text="npy_uri,metadata_uri\n",
        )
        self.assertIn("export PYTHONSAFEPATH=1\ncd /tmp\n", launcher)
        self.assertIn('"$python_bin" -P -m dolma.tokenizer.reshard', launcher)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "launcher.sh"
            path.write_text(launcher)
            self.assertEqual(
                subprocess.run(["bash", "-n", path], check=False).returncode, 0
            )

    def test_materialize_forces_safe_path_for_existing_launchers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            launcher = Path(temp_dir) / "unit.sh"
            launcher.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\npython -m dolma\n"
            )
            payload = _safe_path_launcher_payload(launcher, "123-test-run").decode()
        self.assertIn("export PYTHONSAFEPATH=1", payload)
        self.assertIn("cd /tmp", payload)
        self.assertIn(
            'export DOLMA_STATUS_ROOT="$HOME/dolma3p5-resharding-status/123-test-run"',
            payload,
        )

    def test_materialize_defaults_to_a_complete_oe_other_worker_lifecycle(self):
        args = build_materialize_parser().parse_args(["--all"])
        self.assertEqual(args.project, "oe-other")
        self.assertEqual(args.region, "us-east-1")
        self.assertEqual(args.parallelism, 128)
        self.assertEqual(args.provision_batch_size, 5)
        self.assertEqual(args.provision_batch_delay_seconds, 3.0)
        self.assertEqual(args.bootstrap_parallelism, 32)
        self.assertEqual(args.readiness_poll_seconds, 10)
        self.assertIsNone(args.instance_type)
        self.assertEqual(args.storage_layout, "auto")
        self.assertEqual(args.completion_poll_seconds, 30)
        self.assertFalse(args.verbose)
        self.assertFalse(args.preflight)
        self.assertEqual(args.exclude_category, [])

    def test_materialize_all_can_exclude_completed_categories(self):
        rows = [
            {
                "unit_id": "00000001",
                "leaf_id": "001:00",
                "mix_name": "one:category",
                "category_name": "default",
            },
            {
                "unit_id": "00000002",
                "leaf_id": "002:00",
                "mix_name": "two:category",
                "category_name": "high",
            },
            {
                "unit_id": "00000003",
                "leaf_id": "002:01",
                "mix_name": "two:category",
                "category_name": "mid",
            },
        ]
        args = build_materialize_parser().parse_args(
            ["--all", "--exclude-category", "one:category"]
        )

        label, selected = _select_units(args, rows)

        self.assertEqual(label, "all-except-1-categories")
        self.assertEqual(
            [row["unit_id"] for row in selected], ["00000002", "00000003"]
        )

    def test_materialize_rejects_category_exclusion_without_all(self):
        args = build_materialize_parser().parse_args(
            [
                "--category",
                "one:category",
                "--exclude-category",
                "two:category",
            ]
        )

        with self.assertRaisesRegex(PreparationError, "only be used with --all"):
            _select_units(args, [])

    @patch("scripts.dolma3p5_resharding.materialize.preflight_build")
    def test_materialize_can_run_selection_aware_preflight(self, preflight):
        args = SimpleNamespace(
            profile="read-only",
            region="us-east-1",
            category="dolma3_finemath_v3:finemath::default",
            unit=None,
        )

        selected = [{"unit_id": "00000136"}, {"unit_id": "00000137"}]
        _run_selected_preflight(args, Path("/tmp/build"), selected)

        inline_args = preflight.call_args.args[0]
        self.assertEqual(inline_args.build, Path("/tmp/build"))
        self.assertEqual(inline_args.profile, "read-only")
        self.assertEqual(inline_args.region, "us-east-1")
        self.assertEqual(
            inline_args.category,
            "dolma3_finemath_v3:finemath::default",
        )
        self.assertIsNone(inline_args.unit)
        self.assertEqual(
            inline_args.selected_unit_ids, ("00000136", "00000137")
        )
        self.assertIsNone(inline_args.max_workers)
        self.assertTrue(inline_args.quiet)

        parsed = build_materialize_parser().parse_args(
            ["--all", "--execute", "--preflight"]
        )
        self.assertTrue(parsed.preflight)

    def test_materialize_groups_units_by_planned_worker(self):
        args = build_materialize_parser().parse_args(["--all"])
        rows = [
            {
                "unit_id": "00000000",
                "worker_instance_type": "i4i.2xlarge",
                "worker_storage_layout": "single",
                "worker_vcpus": "8",
            },
            {
                "unit_id": "00000001",
                "worker_instance_type": "i4i.8xlarge",
                "worker_storage_layout": "raid0",
                "worker_vcpus": "32",
            },
        ]

        groups = _planned_worker_groups(args, rows)

        self.assertEqual(
            [(group.instance_type, group.storage_layout) for group in groups],
            [("i4i.2xlarge", "single"), ("i4i.8xlarge", "raid0")],
        )

    def test_materialize_allocates_global_parallelism_across_worker_groups(self):
        groups = [
            PlannedWorkerGroup(
                "i4i.2xlarge",
                "single",
                tuple({"unit_id": f"small-{index}"} for index in range(3)),
            ),
            PlannedWorkerGroup(
                "i4i.8xlarge",
                "raid0",
                tuple({"unit_id": f"large-{index}"} for index in range(5)),
            ),
        ]

        self.assertEqual(_worker_counts_for_groups(groups, 2), [1, 1])
        self.assertEqual(_worker_counts_for_groups(groups, 4), [2, 2])
        self.assertEqual(_worker_counts_for_groups(groups, 8), [3, 5])
        with self.assertRaisesRegex(PreparationError, "must be at least 2"):
            _worker_counts_for_groups(groups, 1)

    def test_materialize_batches_provider_create_requests(self):
        self.assertEqual(_provision_batches(0, 5), ())
        self.assertEqual(_provision_batches(3, 5), (3,))
        self.assertEqual(_provision_batches(12, 5), (5, 5, 2))

    def test_materialize_balances_execution_units_across_worker_slots(self):
        rows = [
            {
                "unit_id": unit_id,
                "estimated_work_uint32_values": str(work),
                "planned_uint32_values": str(work),
            }
            for unit_id, work in (("a", 10), ("b", 9), ("c", 2), ("d", 1))
        ]

        slots = _partition_worker_rows(rows, 2)

        self.assertEqual(
            sorted(
                sum(int(row["estimated_work_uint32_values"]) for row in slot)
                for slot in slots
            ),
            [11, 11],
        )
        self.assertEqual(
            sorted(row["unit_id"] for slot in slots for row in slot),
            ["a", "b", "c", "d"],
        )

    @patch("scripts.dolma3p5_resharding.materialize._retag_cluster_instances")
    @patch("scripts.dolma3p5_resharding.materialize._run_lifecycle_command")
    @patch("scripts.dolma3p5_resharding.materialize._describe_cluster_instances")
    def test_materialize_creates_and_waits_for_the_exact_worker_count(
        self, describe, run, retag
    ):
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=8,
            instance_type="i4i.2xlarge",
            root_storage_type="gp3",
            root_storage_size=200,
            ssh_key_path=None,
            profile=None,
        )
        describe.side_effect = [
            [],
            [
                ClusterInstance("i-second", "running", "i4i.2xlarge", "oe-other"),
                ClusterInstance("i-first", "running", "i4i.2xlarge", "oe-other"),
            ],
        ]

        self.assertEqual(_prepare_workers(args, 2), ["i-first", "i-second"])
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            ["create worker batch 1/1", "wait for workers"],
        )
        create_command = run.call_args_list[0].args[1]
        self.assertEqual(
            create_command[create_command.index("--name") + 1], "dolma3p5-14t"
        )
        self.assertIn("--number", create_command)
        self.assertEqual(create_command[create_command.index("--number") + 1], "2")
        self.assertIn("--detach", create_command)
        wait_command = run.call_args_list[1].args[1]
        self.assertEqual(wait_command[wait_command.index("--name") + 1], "oe-other")
        self.assertEqual(wait_command.count("--instance-id"), 2)
        retag.assert_called_once_with(
            args,
            ["i-first", "i-second"],
            names={
                "i-first": "dolma3p5-14t-0000",
                "i-second": "dolma3p5-14t-0001",
            },
        )

    @patch("scripts.dolma3p5_resharding.materialize._retag_cluster_instances")
    @patch("scripts.dolma3p5_resharding.materialize._run_lifecycle_command")
    @patch("scripts.dolma3p5_resharding.materialize._describe_cluster_instances")
    def test_materialize_launches_large_fleets_in_detached_batches(
        self, describe, run, retag
    ):
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=5,
            provision_batch_size=2,
            provision_batch_delay_seconds=0,
            instance_type="i4i.2xlarge",
            root_storage_type="gp3",
            root_storage_size=200,
            ssh_key_path=None,
            profile=None,
        )
        workers = [
            ClusterInstance(
                f"i-{index}", "pending", "i4i.2xlarge", "oe-other"
            )
            for index in range(5)
        ]
        describe.side_effect = [[], workers[:2], workers[:4], workers]

        self.assertEqual(
            _prepare_workers(args, 5),
            [f"i-{index}" for index in range(5)],
        )

        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                "create worker batch 1/3",
                "create worker batch 2/3",
                "create worker batch 3/3",
                "wait for workers",
            ],
        )
        create_commands = [call.args[1] for call in run.call_args_list[:3]]
        self.assertEqual(
            [command[command.index("--number") + 1] for command in create_commands],
            ["2", "2", "1"],
        )
        self.assertTrue(all("--detach" in command for command in create_commands))
        self.assertEqual(
            [command[command.index("--parallelism") + 1] for command in create_commands],
            ["2", "2", "1"],
        )
        self.assertEqual(retag.call_count, 3)

    @patch("scripts.dolma3p5_resharding.materialize.boto3.Session")
    def test_materialize_retags_project_and_cluster_before_dispatch(self, session):
        client = session.return_value.client.return_value
        client.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-first",
                            "Tags": [
                                {"Key": "project", "Value": "oe-other"},
                                {"Key": "ai2-project", "Value": "oe-other"},
                                {"Key": "cluster", "Value": "dolma3p5-14t"},
                                {"Key": "Name", "Value": "dolma3p5-14t-0000"},
                            ],
                        }
                    ]
                }
            ]
        }
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            profile=None,
        )

        _retag_cluster_instances(
            args,
            ["i-first"],
            names={"i-first": "dolma3p5-14t-0000"},
        )

        self.assertEqual(
            client.create_tags.call_args_list[0],
            call(
                Resources=["i-first"],
                Tags=[
                    {"Key": "project", "Value": "oe-other"},
                    {"Key": "ai2-project", "Value": "oe-other"},
                    {"Key": "cluster", "Value": "dolma3p5-14t"},
                ],
            ),
        )

    @patch("scripts.dolma3p5_resharding.materialize._retag_cluster_instances")
    @patch("scripts.dolma3p5_resharding.materialize._run_lifecycle_command")
    @patch("scripts.dolma3p5_resharding.materialize._describe_cluster_instances")
    def test_materialize_retags_reused_worker_before_resume(self, describe, run, retag):
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=8,
            instance_type="i4i.2xlarge",
            root_storage_type="gp3",
            root_storage_size=200,
            ssh_key_path=None,
            profile=None,
        )
        describe.return_value = [
            ClusterInstance(
                "i-reused",
                "stopped",
                "i4i.2xlarge",
                "oe-other",
                "dolma3p5-14t-0003",
            )
        ]

        self.assertEqual(_prepare_workers(args, 1), ["i-reused"])

        retag.assert_called_once_with(args, ["i-reused"])
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            ["resume workers", "wait for workers"],
        )
        for lifecycle_call in run.call_args_list:
            command = lifecycle_call.args[1]
            self.assertEqual(command[command.index("--name") + 1], "oe-other")
            self.assertEqual(command[command.index("--instance-id") + 1], "i-reused")

    def test_existing_worker_commands_use_project_selector_and_explicit_ids(self):
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=8,
            ssh_key_path=None,
        )
        wait_command = _wait_command(args, ["i-first"])
        map_command = _map_command(args, Path("/tmp/launchers"), ["i-first"])
        log_command = _worker_log_command(args, ["i-first"], "test-run")
        for command in (wait_command, map_command, log_command):
            self.assertEqual(command[command.index("--name") + 1], "oe-other")
            self.assertEqual(command[command.index("--project") + 1], "oe-other")
            self.assertEqual(command[command.index("--instance-id") + 1], "i-first")

    @patch("scripts.dolma3p5_resharding.materialize._describe_cluster_instances")
    def test_materialize_refuses_to_share_a_cluster_with_active_work(self, describe):
        describe.return_value = [
            ClusterInstance("i-busy", "running", "i4i.2xlarge", "oe-other")
        ]
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=8,
            instance_type="i4i.2xlarge",
            root_storage_type="gp3",
            root_storage_size=200,
            ssh_key_path=None,
            profile=None,
        )
        with self.assertRaisesRegex(PreparationError, "already has active"):
            _prepare_workers(args, 1)

    @patch("scripts.dolma3p5_resharding.materialize._retag_cluster_instances")
    @patch("scripts.dolma3p5_resharding.materialize._run_lifecycle_command")
    @patch("scripts.dolma3p5_resharding.materialize._describe_cluster_instances")
    def test_materialize_allows_workers_owned_by_the_same_launch(
        self, describe, run, retag
    ):
        describe.return_value = [
            ClusterInstance("i-owned", "running", "i4i.2xlarge", "oe-other"),
            ClusterInstance("i-large", "stopped", "i4i.8xlarge", "oe-other"),
        ]
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=8,
            instance_type="i4i.8xlarge",
            root_storage_type="gp3",
            root_storage_size=200,
            ssh_key_path=None,
            profile=None,
        )

        self.assertEqual(
            _prepare_workers(args, 1, owned_instance_ids=["i-owned"]),
            ["i-large"],
        )
        retag.assert_called_once_with(args, ["i-large"])
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            ["resume workers", "wait for workers"],
        )

    @patch("scripts.dolma3p5_resharding.materialize._retag_cluster_instances")
    @patch("scripts.dolma3p5_resharding.materialize._run_lifecycle_command")
    @patch("scripts.dolma3p5_resharding.materialize._describe_cluster_instances")
    def test_materialize_can_defer_reused_worker_resume(
        self, describe, run, retag
    ):
        describe.return_value = [
            ClusterInstance("i-reused", "stopped", "i4i.2xlarge", "oe-other")
        ]
        args = SimpleNamespace(
            cluster="dolma3p5-14t",
            project="oe-other",
            region="us-east-1",
            parallelism=2,
            instance_type="i4i.2xlarge",
            root_storage_type="gp3",
            root_storage_size=200,
            ssh_key_path=None,
            profile=None,
        )
        resume_ids = []

        self.assertEqual(
            _prepare_workers(
                args,
                1,
                wait_for_ready=False,
                deferred_resume_ids=resume_ids,
            ),
            ["i-reused"],
        )

        self.assertEqual(resume_ids, ["i-reused"])
        retag.assert_called_once_with(args, ["i-reused"])
        run.assert_not_called()

    @patch("scripts.dolma3p5_resharding.materialize._pause_workers_after_failure")
    @patch("scripts.dolma3p5_resharding.materialize._verify_materialized_units")
    @patch("scripts.dolma3p5_resharding.materialize._wait_for_workers_to_stop")
    @patch("scripts.dolma3p5_resharding.materialize._bootstrap_and_dispatch_worker")
    @patch("scripts.dolma3p5_resharding.materialize._ready_worker_ids")
    @patch("scripts.dolma3p5_resharding.materialize._stage_worker_assignments")
    @patch("scripts.dolma3p5_resharding.materialize._resume_command", return_value=["resume"])
    @patch("scripts.dolma3p5_resharding.materialize._run_lifecycle_command")
    @patch("scripts.dolma3p5_resharding.materialize._prepare_workers")
    def test_materialize_dispatches_workers_as_each_becomes_ready(
        self,
        prepare,
        run,
        resume_command,
        stage_assignments,
        ready_worker_ids,
        bootstrap,
        monitor,
        verify,
        pause,
    ):
        events = []

        def prepare_group(
            group_args,
            worker_count,
            *,
            owned_instance_ids=(),
            wait_for_ready=True,
            delay_after_last_batch=False,
            deferred_resume_ids=None,
        ):
            instance_id = (
                "i-small" if group_args.instance_type == "i4i.2xlarge" else "i-large"
            )
            events.append(
                (
                    "prepare",
                    group_args.instance_type,
                    tuple(owned_instance_ids),
                    wait_for_ready,
                    delay_after_last_batch,
                )
            )
            if deferred_resume_ids is not None:
                deferred_resume_ids.append(instance_id)
            return [instance_id]

        prepare.side_effect = prepare_group
        run.side_effect = lambda stage, command, **kwargs: events.append(("run", stage))
        selected = [
            {
                "unit_id": "small",
                "planned_uint32_values": "10",
                "estimated_work_uint32_values": "20",
            },
            {
                "unit_id": "large",
                "planned_uint32_values": "20",
                "estimated_work_uint32_values": "40",
            },
        ]
        groups = [
            MaterializationGroup(
                args=SimpleNamespace(instance_type="i4i.2xlarge"),
                rows=(selected[0],),
                script_dir=Path("/tmp/small"),
                worker_count=1,
            ),
            MaterializationGroup(
                args=SimpleNamespace(instance_type="i4i.8xlarge"),
                rows=(selected[1],),
                script_dir=Path("/tmp/large"),
                worker_count=1,
            ),
        ]
        stage_assignments.side_effect = [
            (
                WorkerAssignment(
                    groups[0], groups[0].rows, Path("/tmp/small-assignment")
                ),
            ),
            (
                WorkerAssignment(
                    groups[1], groups[1].rows, Path("/tmp/large-assignment")
                ),
            ),
        ]
        ready_worker_ids.side_effect = [{"i-small"}, {"i-large"}]

        def dispatch_worker(
            dispatch_args, instance_id, assignment, stages, stage_lock, console
        ):
            events.append(("dispatch", instance_id, assignment.rows[0]["unit_id"]))
            with stage_lock:
                stages[instance_id] = "materializing"

        bootstrap.side_effect = dispatch_worker

        def monitor_workers(*monitor_args, **monitor_kwargs):
            self.assertTrue(monitor_kwargs["all_dispatched"].wait(timeout=2))
            events.append(("monitor-complete",))

        monitor.side_effect = monitor_workers
        verify.side_effect = lambda *verify_args, **verify_kwargs: events.append(
            ("verify",)
        )
        args = SimpleNamespace(
            verbose=True,
            bootstrap_parallelism=2,
            readiness_poll_seconds=1,
        )

        _execute_materialization_groups(args, groups, selected, "test-run")

        self.assertEqual(
            events[:2],
            [
                ("prepare", "i4i.2xlarge", (), False, True),
                ("prepare", "i4i.8xlarge", ("i-small",), False, False),
            ],
        )
        resume_command.assert_called_once_with(
            args, ["i-small", "i-large"], detach=True
        )
        self.assertEqual(
            [event for event in events if event[0] == "dispatch"],
            [("dispatch", "i-small", "small"), ("dispatch", "i-large", "large")],
        )
        self.assertEqual(
            ready_worker_ids.call_args_list,
            [call(args, ["i-large", "i-small"]), call(args, ["i-large"])],
        )
        monitor.assert_called_once()
        verify.assert_called_once_with(args, selected)
        pause.assert_not_called()

    @patch("scripts.dolma3p5_resharding.materialize._pause_workers_after_failure")
    @patch("scripts.dolma3p5_resharding.materialize._stage_worker_assignments")
    @patch("scripts.dolma3p5_resharding.materialize._prepare_workers")
    def test_materialize_cleans_up_prepared_groups_if_later_provisioning_fails(
        self, prepare, stage_assignments, pause
    ):
        prepare.side_effect = [["i-small"], PreparationError("create failed")]
        args = SimpleNamespace(verbose=False)
        selected = [{"unit_id": "small"}, {"unit_id": "large"}]
        groups = [
            MaterializationGroup(
                args=SimpleNamespace(instance_type="i4i.2xlarge"),
                rows=(selected[0],),
                script_dir=Path("/tmp/small"),
                worker_count=1,
            ),
            MaterializationGroup(
                args=SimpleNamespace(instance_type="i4i.8xlarge"),
                rows=(selected[1],),
                script_dir=Path("/tmp/large"),
                worker_count=1,
            ),
        ]
        stage_assignments.side_effect = [
            (WorkerAssignment(groups[0], groups[0].rows, Path("/tmp/small-worker")),),
            (WorkerAssignment(groups[1], groups[1].rows, Path("/tmp/large-worker")),),
        ]

        with self.assertRaisesRegex(PreparationError, "create failed"):
            _execute_materialization_groups(args, groups, selected, "test-run")

        pause.assert_called_once_with(args, ["i-small"])

    def test_region_defaults_to_us_east_1_and_allows_override(self):
        self.assertEqual(normalize_region(None), DEFAULT_REGION)
        self.assertEqual(normalize_region(""), DEFAULT_REGION)
        self.assertEqual(normalize_region("  "), DEFAULT_REGION)
        self.assertEqual(normalize_region(" us-west-2 "), "us-west-2")

    def test_source_layout_preserves_exact_source_directory(self):
        self.assertEqual(
            _source_relative_directory(
                "preprocessed/dolma3-0625/v0.1-official/allenai/"
                "dolma3-tokenizer/finemath-3plus/000000.npy"
            ),
            "dolma3-0625/v0.1-official/allenai/dolma3-tokenizer/finemath-3plus",
        )
        self.assertEqual(
            _source_relative_directory(
                "preprocessed/cc_all_dressed/all_dressed_v5/topic/health/"
                "vigintile_0016/allenai/dolma2-tokenizer/000000.npy"
            ),
            "cc_all_dressed/all_dressed_v5/topic/health/"
            "vigintile_0016/allenai/dolma2-tokenizer",
        )

    def test_category_output_directory_replaces_varying_lower_group(self):
        objects = [
            {
                "key": (
                    "preprocessed/cc_all_dressed/all_dressed_v5/topic/health/"
                    f"vigintile_{index:04d}/allenai/dolma2-tokenizer/000000.npy"
                )
            }
            for index in range(16, 20)
        ]
        self.assertEqual(
            _category_output_directory(objects, "high"),
            "cc_all_dressed/all_dressed_v5/topic/health/high/allenai/dolma2-tokenizer",
        )

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
                            "paths": ["dolma3p5_pool/catalog-source/topic/allenai/tokenizer/*.npy"],
                            "repetition_factor": -1.0,
                        },
                        {
                            "name": "dropped",
                            "weight": 0.0,
                            "paths": ["dolma3p5_pool/catalog-source/topic/vigintile_0000/allenai/tokenizer/*.npy"],
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
                            "paths": ["preprocessed/direct-source/allenai/tokenizer/*.npy"],
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
        self.settings = self.root / "settings.yaml"
        self.settings.write_text(
            yaml.safe_dump(
                {
                    "worker_instance_grid": [
                        {
                            "instance_type": "i4i.32xlarge",
                            "vcpus": 128,
                            "local_nvme_devices": 8,
                            "local_nvme_bytes": 30_000_000_000_000,
                            "max_estimated_work_uint32_values": None,
                        }
                    ]
                },
                sort_keys=False,
            )
        )
        self.build = self.root / "build"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _plan(self):
        plan_build(
            argparse.Namespace(
                mix=self.mix,
                catalog=self.catalog,
                settings=self.settings,
                output=self.build,
            )
        )

    def _write_inventory(self):
        listing = self.root / "listing.jsonl"
        with listing.open("x") as f:
            for uri, size in self._inventory_objects().items():
                f.write(json.dumps(self._inventory_record(uri, size)) + "\n")
        phase = self.build / "01-plan/inventory"
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
                    stdout.write(json.dumps(self._inventory_record(uri, size)) + "\n")
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
        collector = json.loads(
            (self.build / "01-plan/inventory/collector.json").read_text()
        )
        self.assertEqual(collector["collector"], "s5cmd")
        self.assertEqual(collector["output_records"], 8)
        self.assertTrue((self.build / "01-plan/inventory/raw-listings.jsonl").is_file())
        status_output = output.getvalue()
        self.assertIn("[inventory 1/4] Bulk listing:", status_output)
        self.assertIn("[inventory 1/4] Running:", status_output)
        self.assertIn("[inventory 2/4] Resolving required objects for", status_output)
        self.assertIn("[inventory 3/4] Validation:", status_output)
        self.assertIn("[inventory 3/4] Estimated source tokens:", status_output)
        self.assertIn("[inventory 4/4] PASS:", status_output)

    def test_inventory_requires_s5cmd_without_replacing_existing_artifacts(self):
        self._plan()
        inventory_phase = self.build / "01-plan/inventory"
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

    def test_catalog_resolves_yaml_wildcard_against_live_listing(self):
        self._plan()
        objects = self._inventory_objects() | {
            "s3://ai2-llm/preprocessed/catalog-source/topic/allenai/tokenizer/0001.npy": 100,
            "s3://ai2-llm/preprocessed/catalog-source/topic/allenai/tokenizer/0001.csv.gz": 12,
        }
        listing = self.root / "expanded-listing.jsonl"
        with listing.open("x") as handle:
            for uri, size in objects.items():
                handle.write(json.dumps(self._inventory_record(uri, size)) + "\n")
        listed, errors = _parse_s5cmd_jsonl(listing)
        self.assertFalse(errors)
        phase = self.build / "01-plan/inventory"
        phase.mkdir()
        summary = _finalize_inventory(
            self.build,
            phase,
            listed,
            client=MagicMock(),
            max_workers=1,
        )
        with (phase / "required-objects.csv").open() as handle:
            required = list(csv.DictReader(handle))
        catalog_topic = [
            row
            for row in required
            if row["leaf_id"] == "000:00" and row["category_name"] == "default"
        ]
        self.assertEqual(
            [row["key"].rsplit("/", 1)[-1] for row in catalog_topic],
            ["0000.npy", "0001.npy"],
        )
        self.assertEqual(summary["source_uint32_values"], 675)

    def test_sampling_rate_limit_blocks_proposal(self):
        self._plan()
        build_manifest_path = self.build / "build.json"
        build_manifest = json.loads(build_manifest_path.read_text())
        build_manifest["settings"]["maximum_expected_upsample_rate"] = 7.2
        build_manifest_path.write_text(json.dumps(build_manifest))
        listing = self.root / "limited-listing.jsonl"
        with listing.open("x") as handle:
            for uri, size in self._inventory_objects().items():
                handle.write(json.dumps(self._inventory_record(uri, size)) + "\n")
        listed, errors = _parse_s5cmd_jsonl(listing)
        self.assertFalse(errors)
        phase = self.build / "01-plan/inventory"
        phase.mkdir()
        with self.assertRaisesRegex(PreparationError, "Inventory validation failed"):
            _finalize_inventory(
                self.build,
                phase,
                listed,
                client=MagicMock(),
                max_workers=1,
            )
        summary = json.loads((phase / "inventory-summary.json").read_text())
        self.assertEqual(summary["sampling_rate_limit_failures"], 3)
        with (phase / "sampling-rate-audit.csv").open() as handle:
            audit = list(csv.DictReader(handle))
        self.assertEqual(
            sum(row["status"] == "above_expected_maximum" for row in audit),
            3,
        )
        inventory_report = (phase / "report.html").read_text()
        self.assertIn("Source-size consistency check failed", inventory_report)
        self.assertIn("Proposal generation is blocked", inventory_report)
        with self.assertRaisesRegex(PreparationError, "blocking validation failures"):
            propose_configs(
                argparse.Namespace(
                    build=self.build,
                    destination_root="s3://test-bucket/new-datasets/dolma3p5",
                    local_temp_root=str(self.root / "temp-base"),
                    max_unit_working_bytes=20_000_000_000_000,
                )
            )

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
            (self.build / "01-plan/resolution/plots/resolution-counts.svg").exists()
        )
        self.assertFalse(
            (self.build / "01-plan/resolution/plot-data/resolution-counts.csv").exists()
        )

    def test_plan_command_builds_resolution_inventory_and_execution_together(self):
        from scripts.dolma3p5_resharding import plan as plan_command

        output = self.root / "combined-plan"
        with (
            patch.object(plan_command.shutil, "which", return_value="/usr/bin/s5cmd"),
            patch.object(plan_command, "plan_build") as build_paths,
            patch.object(plan_command, "collect_inventory") as inventory,
            patch.object(plan_command, "propose_configs") as execution,
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
                    "--destination-root",
                    "s3://test-bucket/new-datasets/dolma3p5",
                    "--local-temp-root",
                    "/mnt/dolma/dolma3p5-resharding",
                    "--max-unit-working-bytes",
                    "1500000000000",
                ],
            ),
        ):
            plan_command.main()

        build_paths.assert_called_once()
        inventory.assert_called_once()
        execution.assert_called_once()
        inventory_args = inventory.call_args.args[0]
        self.assertEqual(inventory_args.build, output)
        self.assertEqual(inventory_args.profile, "read-only")
        execution_args = execution.call_args.args[0]
        self.assertEqual(execution_args.build, output)
        self.assertEqual(
            execution_args.destination_root,
            "s3://test-bucket/new-datasets/dolma3p5",
        )
        self.assertEqual(
            execution_args.local_temp_root, Path("/mnt/dolma/dolma3p5-resharding")
        )
        self.assertEqual(execution_args.max_unit_working_bytes, 1_500_000_000_000)

    def test_catalog_paths_are_decoded_to_literal_s3_keys(self):
        catalog = self.root / "encoded-catalog.csv"
        catalog.write_text("ai2-llm,preprocessed/the-stack-v2/C%2B%2B/0000.npy\n")

        rows = _load_catalog(catalog)

        self.assertEqual(rows[0]["key"], "preprocessed/the-stack-v2/C++/0000.npy")

    def test_execution_units_support_exact_category_and_unit_selection(self):
        rows = [
            {
                "unit_id": "finemath-0001",
                "leaf_id": "060:00",
                "mix_name": "dolma3_finemath_v3:finemath",
                "category_name": "default",
            },
            {
                "unit_id": "finemath-0002",
                "leaf_id": "060:00",
                "mix_name": "dolma3_finemath_v3:finemath",
                "category_name": "default",
            },
            {
                "unit_id": "stack-0001",
                "leaf_id": "061:00",
                "mix_name": "the-stack-v2:Python",
                "category_name": "high",
            },
        ]
        selected_mix = _filter_execution_units(
            rows, category="dolma3_finemath_v3:finemath"
        )
        selected_leaf = _filter_execution_units(rows, category="060:00")
        selected_full = _filter_execution_units(
            rows,
            category="the-stack-v2:Python::high",
        )
        selected_unit = _filter_execution_units(rows, unit="finemath-0002")

        self.assertEqual(
            [row["unit_id"] for row in selected_mix], ["finemath-0001", "finemath-0002"]
        )
        self.assertEqual(
            _unit_selection_digest(selected_mix), _unit_selection_digest(selected_leaf)
        )
        self.assertEqual([row["unit_id"] for row in selected_full], ["stack-0001"])
        self.assertEqual([row["unit_id"] for row in selected_unit], ["finemath-0002"])

    def test_s5cmd_listing_commands_use_literal_s3_keys(self):
        mix = self.root / "encoded-mix.yaml"
        catalog = self.root / "encoded-command-catalog.csv"
        build = self.root / "encoded-command-build"
        yaml_path = (
            "dolma3p5_pool/the-stack-v2/C++/quality_p95/allenai/dolma2-tokenizer/*.npy"
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

        commands = (build / "01-plan/resolution/bulk-listing-commands.txt").read_text()
        self.assertIn("/C++/quality_p95/", commands)
        self.assertNotIn("%2B", commands)

    def test_end_to_end_preparation_phases_are_replaceable(self):
        mix_before = self.mix.read_bytes()
        catalog_before = self.catalog.read_bytes()
        self._plan()
        plan_report = (self.build / "01-plan/resolution/report.html").read_text()
        self.assertIn("Dolma 3.5 Target Allocation and Source Path Plan", plan_report)
        self.assertIn("Materialized output target:", plan_report)
        self.assertNotIn("S3 source volume:", plan_report)
        self.assertIn('data-detail="plan-family-detail-', plan_report)
        self.assertIn('class="subcategory-row', plan_report)
        self.assertIn('class="subcategory-detail"', plan_report)
        self.assertIn('class="category-grid"', plan_report)
        self.assertIn("matched NPY", plan_report)
        self.assertIn("topic", plan_report)
        self.assertIn(
            "s3://ai2-llm/preprocessed/catalog-source/topic/allenai/tokenizer/0000.npy",
            plan_report,
        )
        self.assertIn(
            "s3://ai2-llm/preprocessed/direct-source/allenai/tokenizer/*.npy",
            plan_report,
        )
        self.assertNotIn(
            "dolma3p5_pool/catalog-source/topic/allenai/tokenizer/*.npy",
            plan_report,
        )
        with (self.build / "01-plan/resolution/normalized-paths.csv").open() as f:
            self.assertNotIn("resolution_route", csv.DictReader(f).fieldnames)
        with (self.build / "01-plan/resolution/listing-plan.csv").open() as f:
            self.assertNotIn("resolution_routes", csv.DictReader(f).fieldnames)
        plan_target_plot = (self.build / "01-plan/resolution/plots/target-mix.svg").read_text()
        self.assertIn("Total target: 14T tokens (14,000,000,000,000)", plan_target_plot)
        self.assertIn("50.00% · 7T tokens", plan_target_plot)
        self.assertNotIn('text-anchor="end"', plan_target_plot)
        with (self.build / "01-plan/resolution/corrections.csv").open() as f:
            corrections = list(csv.DictReader(f))
        self.assertEqual(len(corrections), 1)
        self.assertIn("quality_p95", corrections[0]["path"])
        with (self.build / "01-plan/resolution/direct-s3-patterns.csv").open() as f:
            direct = list(csv.DictReader(f))
        self.assertEqual(len(direct), 1)
        finder_metadata = self.build / ".DS_Store"
        finder_metadata.write_bytes(b"preserve benign metadata")
        legacy_proposal = self.build / "03-proposal"
        legacy_proposal.mkdir()
        (legacy_proposal / "old-generated-artifact.txt").write_text("replaceable")
        self._plan()
        self.assertEqual(self.mix.read_bytes(), mix_before)
        self.assertEqual(self.catalog.read_bytes(), catalog_before)
        self.assertEqual(finder_metadata.read_bytes(), b"preserve benign metadata")
        self.assertFalse(legacy_proposal.exists())

        self._write_inventory()
        inventory_summary = json.loads((self.build / "01-plan/inventory/inventory-summary.json").read_text())
        self.assertEqual(inventory_summary["source_count"], 3)
        self.assertEqual(inventory_summary["source_family_count"], 3)
        self.assertEqual(inventory_summary["subcategory_count"], 3)
        self.assertEqual(inventory_summary["category_count"], 4)
        self.assertEqual(inventory_summary["lower_group_count"], 4)
        self.assertEqual(inventory_summary["source_uint32_values"], 650)
        self.assertEqual(inventory_summary["token_delta"], 13_999_999_999_350)
        self.assertGreater(inventory_summary["sampling_ratio"], 1)
        self.assertIn("upsample", inventory_summary["sampling_rate"])
        self.assertEqual(inventory_summary["details_artifact"], "inventory-details.json")
        inventory_details = json.loads((self.build / "01-plan/inventory/inventory-details.json").read_text())
        self.assertEqual(inventory_details["source_uint32_values"], 650)
        self.assertEqual(inventory_details["source_family_count"], 3)
        self.assertEqual(inventory_details["subcategory_count"], 3)
        catalog_source = next(
            source for source in inventory_details["sources"] if source["mix_name"] == "catalog-source:topic"
        )
        self.assertEqual(catalog_source["source_uint32_values"], 150)
        self.assertEqual(catalog_source["source_family"], "catalog-source")
        self.assertEqual(catalog_source["subcategory_name"], "topic")
        dropped_category = next(
            category for category in catalog_source["categories"] if category["category_name"] == "dropped"
        )
        self.assertFalse(dropped_category["active"])
        self.assertEqual(dropped_category["source_uint32_values"], 50)
        self.assertAlmostEqual(dropped_category["source_percent_of_parent"], 100 / 3)
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
        inventory_plot = (self.build / "01-plan/inventory/plots/available-by-mix.svg").read_text()
        self.assertIn("% ·", inventory_plot)
        self.assertIn("tokens", inventory_plot)
        self.assertFalse((self.build / "01-plan/inventory/plot-data/object-size-bins.csv").exists())
        self.assertFalse((self.build / "01-plan/inventory/plots/object-size-histogram.svg").exists())
        inventory_report = (self.build / "01-plan/inventory/report.html").read_text()
        self.assertIn('data-detail="inventory-family-detail-', inventory_report)
        self.assertIn('class="subcategory-row', inventory_report)
        self.assertIn('class="subcategory-detail"', inventory_report)
        self.assertIn("const setAccordionState", inventory_report)
        self.assertIn("button.getAttribute('aria-expanded') !== 'true'", inventory_report)
        self.assertIn('class="comparison-bars"', inventory_report)
        self.assertIn('class="category-metrics"', inventory_report)
        self.assertNotIn('class="category-counts"', inventory_report)
        self.assertIn('class="path-stat"', inventory_report)
        self.assertIn('<span class="path-stat">1 file', inventory_report)
        self.assertNotIn('<span class="path-stat">1 NPY', inventory_report)
        self.assertIn('class="path-metric"', inventory_report)
        self.assertIn('class="path-detail-metrics"', inventory_report)
        self.assertIn('class="path-detail-uri"', inventory_report)
        self.assertIn(
            "s3://ai2-llm/preprocessed/catalog-source/topic/allenai/tokenizer/0000.npy",
            inventory_report,
        )
        self.assertNotIn(
            "dolma3p5_pool/catalog-source/topic/allenai/tokenizer/*.npy",
            inventory_report,
        )
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
        sampling_path = self.build / "01-plan/inventory/plot-data/sampling-by-lower-group.csv"
        with sampling_path.open() as f:
            sampling_paths = list(csv.DictReader(f))
        dropped = next(row for row in sampling_paths if row["lower_group"] == "vigintile_0000")
        self.assertEqual(dropped["original_uint32_values"], "50")
        self.assertEqual(dropped["implied_target_uint32_values"], "0")
        self.assertIn("downsample", dropped["sampling_rate"])
        with (self.build / "01-plan/inventory/required-objects.csv").open() as f:
            self.assertNotIn("resolution_route", csv.DictReader(f).fieldnames)
        propose_configs(
            argparse.Namespace(
                build=self.build,
                destination_root="s3://test-bucket/new-datasets/dolma3p5",
                local_temp_root=str(self.root / "temp-base"),
                max_unit_working_bytes=20_000_000_000_000,
            )
        )
        self.assertEqual(
            {path.name for path in (self.build / "01-plan").iterdir()},
            {"resolution", "inventory", "execution", "report.html"},
        )
        self.assertFalse((self.build / "01-plan/resolution/report.html").exists())
        self.assertFalse((self.build / "01-plan/inventory/report.html").exists())
        self.assertFalse((self.build / "01-plan/execution/report.html").exists())
        self.assertFalse((self.build / "02-inventory").exists())
        self.assertFalse((self.build / "03-proposal").exists())
        configs = list((self.build / "01-plan/execution/config").glob("*.yaml"))
        self.assertEqual(len(configs), 4)
        launcher_scripts = list((self.build / "01-plan/execution/launcher-scripts").glob("*.sh"))
        self.assertEqual(len(launcher_scripts), 4)
        self.assertTrue(all(path.stat().st_mode & 0o100 for path in launcher_scripts))
        self.assertTrue(
            all(
                '"$python_bin" -P -m dolma.tokenizer.reshard' in path.read_text()
                and "export PYTHONSAFEPATH=1" in path.read_text()
                and "cd /tmp" in path.read_text()
                for path in launcher_scripts
            )
        )
        self.assertTrue(
            all(
                "dolma3p5-resharding-status" in path.read_text()
                and "printf 'running" in path.read_text()
                and "printf 'succeeded" in path.read_text()
                and "printf 'failed" in path.read_text()
                for path in launcher_scripts
            )
        )
        self.assertTrue(all("RESHARDING_MANIFEST_SCHEMA_VERSION" in path.read_text() for path in launcher_scripts))
        for path in launcher_scripts:
            self.assertEqual(
                subprocess.run(["bash", "-n", path], check=False).returncode,
                0,
            )
        with (self.build / "01-plan/execution/config-index.csv").open() as f:
            execution_units = list(csv.DictReader(f))
        self.assertEqual(
            [row["unit_id"] for row in execution_units],
            [f"{index:0{EXECUTION_UNIT_INDEX_WIDTH}d}" for index in range(len(execution_units))],
        )
        self.assertTrue(
            all(
                int(row["estimated_peak_local_bytes"]) <= int(row["max_unit_working_bytes"])
                for row in execution_units
            )
        )
        self.assertTrue(
            all(
                1 <= int(row["planned_output_shard_count"]) <= 8
                and int(row["max_num_files"]) == int(row["planned_output_shard_count"])
                and int(row["average_output_shard_bytes"])
                == (
                    int(row["output_npy_bytes"])
                    + int(row["planned_output_shard_count"])
                    - 1
                )
                // int(row["planned_output_shard_count"])
                for row in execution_units
            )
        )
        self.assertTrue(all(row["worker_instance_type"] == "i4i.32xlarge" for row in execution_units))
        self.assertTrue(all(int(row["max_workers"]) == 32 for row in execution_units))
        dataset_layout = json.loads((self.build / "01-plan/execution/dataset-layout.json").read_text())
        self.assertEqual(dataset_layout["category_count"], 3)
        self.assertEqual(dataset_layout["execution_unit_count"], 4)
        self.assertEqual(dataset_layout["schema_version"], 1)
        self.assertRegex(
            dataset_layout["dataset_root"],
            r"^s3://test-bucket/new-datasets/dolma3p5/dolma3p5-14t-[0-9a-f]{12}$",
        )
        self.assertEqual(dataset_layout["layout"], "build-scoped-category-output-v1")
        self.assertEqual(dataset_layout["execution_unit_index_width"], EXECUTION_UNIT_INDEX_WIDTH)
        self.assertEqual(dataset_layout["execution_unit_id_width"], EXECUTION_UNIT_INDEX_WIDTH)
        self.assertEqual(
            dataset_layout["planned_output_shard_count"],
            sum(int(row["planned_output_shard_count"]) for row in execution_units),
        )
        self.assertEqual(
            dataset_layout["planned_output_file_count"],
            dataset_layout["planned_output_shard_count"] * 2,
        )
        self.assertEqual(dataset_layout["target_output_shard_bytes"], 64 * 1024**3)
        self.assertEqual(dataset_layout["max_output_shards_per_unit"], 8)
        self.assertEqual(_validate_execution_layout(self.build), dataset_layout)
        runtime_requirements = json.loads((self.build / "01-plan/execution/runtime-requirements.json").read_text())
        self.assertEqual(runtime_requirements["required_resharding_manifest_schema_version"], 2)
        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            self.assertFalse(config["allow_existing_destination"])
            self.assertEqual(config["s5cmd_download_concurrency"], 32)
            self.assertTrue(config["source_manifests"])
            parsed = ReshardingConfig.from_file(config_path)
            self.assertEqual(len(parsed.source_manifests), 1)
            self.assertTrue(Path(parsed.source_manifests[0].manifest).is_file())
            destination = config["destination_prefix"]
            self.assertRegex(
                destination,
                r"^s3://test-bucket/new-datasets/dolma3p5/dolma3p5-14t-[0-9a-f]{12}/",
            )
            self.assertRegex(destination, rf"/[0-9]{{{EXECUTION_UNIT_INDEX_WIDTH}}}$")
            self.assertNotIn("/categories/", destination)
            self.assertNotIn("/unit-", destination)
        for row in execution_units:
            expected_suffix = f"/{row['source_layout_prefix']}/{row['destination_index']}"
            self.assertIn(expected_suffix, row["destination_prefix"])
        for manifest_path in (self.build / "01-plan/execution/manifests").glob("*.csv"):
            with manifest_path.open() as handle:
                fields = csv.DictReader(handle).fieldnames
            self.assertIn("partial_target_uint32_values", fields)
            self.assertIn("selection_seed", fields)
            self.assertIn("selection_algorithm", fields)
        for plot in (self.build / "01-plan/execution/plots").glob("*.svg"):
            ElementTree.parse(plot)
        self.assertFalse(
            (self.build / "01-plan/execution/plot-data/category-execution-unit-distribution.csv").exists()
        )
        self.assertFalse((self.build / "01-plan/execution/plot-data/most-split-categories.csv").exists())
        self.assertFalse((self.build / "01-plan/execution/plots/execution-units-per-category.svg").exists())
        self.assertFalse((self.build / "01-plan/execution/plot-data/object-size-bins.csv").exists())
        self.assertFalse((self.build / "01-plan/execution/plots/object-size-histogram.svg").exists())
        self.assertFalse((self.build / "01-plan/execution/plot-data/source-vs-target.csv").exists())
        self.assertFalse((self.build / "01-plan/execution/plots/source-vs-target.svg").exists())
        proposal_target_plot = (self.build / "01-plan/execution/plots/target-mix.svg").read_text()
        self.assertIn("50.00% · 7T tokens", proposal_target_plot)
        self.assertFalse((self.build / "01-plan/execution/plot-data/target-vs-proposed.csv").exists())
        self.assertFalse((self.build / "01-plan/execution/plots/target-vs-proposed.svg").exists())
        self.assertFalse((self.build / "01-plan/execution/plot-data/upsampling-pressure.csv").exists())
        self.assertFalse((self.build / "01-plan/execution/plots/upsampling-pressure.svg").exists())
        proposal_report = (self.build / "01-plan/report.html").read_text()
        self.assertIn("Source inventory &amp; sampling", proposal_report)
        self.assertIn("Materialization execution", proposal_report)
        self.assertIn('role="tablist"', proposal_report)
        self.assertIn('id="source-report-document"', proposal_report)
        self.assertIn('id="execution-report-document"', proposal_report)
        self.assertIn('<base href="inventory/">', proposal_report)
        self.assertIn('<base href="execution/">', proposal_report)
        self.assertIn("Source Inventory and Sampling Plan", proposal_report)
        self.assertIn("Materialization Execution Proposal", proposal_report)
        self.assertIn('class="execution-metrics"', proposal_report)
        self.assertIn("Execution units", proposal_report)
        self.assertIn("Output shards", proposal_report)
        self.assertIn("Output files", proposal_report)
        self.assertNotIn("Units using document selection", proposal_report)
        self.assertIn("Planned worker fleet", proposal_report)
        self.assertIn('class="worker-grid"', proposal_report)
        self.assertIn("Local NVMe", proposal_report)
        self.assertIn("Median unit", proposal_report)
        self.assertIn("P95 unit", proposal_report)
        self.assertIn("Largest unit", proposal_report)
        self.assertIn("Median output shard", proposal_report)
        self.assertIn("P95 output shard", proposal_report)
        self.assertNotIn("concurrent units on the same worker", proposal_report)
        self.assertEqual(
            proposal_report.count('class="execution-unit"'),
            dataset_layout["execution_unit_count"],
        )
        self.assertIn('class="disk-breakdown"', proposal_report)
        self.assertIn("Source download", proposal_report)
        self.assertIn("Materialized output", proposal_report)
        self.assertIn("Selection indexes", proposal_report)
        self.assertIn("Source shard downloads", proposal_report)
        self.assertIn("Source shards using document selection", proposal_report)
        self.assertIn("Reshard concurrency", proposal_report)
        self.assertIn("Output layout", proposal_report)
        self.assertIn("shards ·", proposal_report)
        self.assertIn("Only categories with multiple units", proposal_report)
        self.assertIn('href="config/', proposal_report)
        self.assertIn('href="manifests/', proposal_report)
        self.assertIn('href="launcher-scripts/', proposal_report)
        self.assertIn("/new-datasets/dolma3p5/", proposal_report)
        execution_report = proposal_report.split('<template id="execution-report-document">', 1)[1]
        self.assertNotIn("Overall sampling", execution_report)
        self.assertNotIn("full copies per object", execution_report)
        self.assertNotIn("NPYs repeated", execution_report)
        self.assertNotIn("total object uses", execution_report)
        self.assertNotIn('class="path-use-summary"', execution_report)
        self.assertNotIn('class="subcategory-detail"', execution_report)
        with (self.build / "01-plan/execution/category-allocation.csv").open() as f:
            allocation_rows = list(csv.DictReader(f))
        catalog_allocation = next(row for row in allocation_rows if row["mix_name"] == "catalog-source:topic")
        self.assertEqual(catalog_allocation["available_uint32_values"], "100")
        self.assertEqual(
            int(catalog_allocation["planned_uint32_values"]) - int(catalog_allocation["available_uint32_values"]),
            int(catalog_allocation["token_change_from_original"]),
        )
        self.assertEqual(
            catalog_allocation["minimum_repetition"],
            catalog_allocation["maximum_repetition"],
        )
        self.assertGreater(int(catalog_allocation["maximum_repetition"]), 1)
        self.assertEqual(int(catalog_allocation["target_residual_uint32_values"]), 0)
        with (self.build / "01-plan/execution/plot-data/proposed-sampling-by-lower-group.csv").open() as f:
            proposed_paths = list(csv.DictReader(f))
        dropped_proposal = next(row for row in proposed_paths if row["lower_group"] == "vigintile_0000")
        self.assertEqual(dropped_proposal["original_uint32_values"], "50")
        self.assertEqual(dropped_proposal["proposed_uint32_values"], "0")
        self.assertEqual(dropped_proposal["token_change_from_original"], "-50")
        self.assertEqual(dropped_proposal["minimum_repetition"], "0")
        self.assertEqual(dropped_proposal["maximum_repetition"], "0")
        self.assertEqual(dropped_proposal["dropped_object_count"], "1")
        self.assertEqual(dropped_proposal["repeated_object_count"], "0")
        proposal_summary = json.loads((self.build / "01-plan/execution/proposal-summary.json").read_text())
        self.assertEqual(proposal_summary["source_uint32_values"], 650)
        self.assertEqual(
            proposal_summary["token_change_from_source"],
            proposal_summary["planned_uint32_values"] - 650,
        )
        self.assertEqual(proposal_summary["target_residual_uint32_values"], 0)
        self.assertEqual(
            proposal_summary["planned_output_shard_count"],
            dataset_layout["planned_output_shard_count"],
        )
        self.assertEqual(
            proposal_summary["planned_output_file_count"],
            dataset_layout["planned_output_file_count"],
        )
        self.assertEqual(
            proposal_summary["document_selection_algorithm"],
            "document_hash_bucket_v1",
        )

        with (self.build / "01-plan/inventory/normalized-s3-inventory.csv").open() as f:
            inventory_rows = [row for row in csv.DictReader(f) if row["required"] == "true"]
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
        self.assertFalse((self.build / "02-preflight/plots").exists())
        self.assertFalse((self.build / "02-preflight/plot-data").exists())
        self.assertFalse((self.build / "02-preflight/report.html").exists())

        with (self.build / "01-plan/execution/config-index.csv").open() as f:
            config_index = list(csv.DictReader(f))
        output_by_prefix = {}
        for row in config_index:
            destination = row["destination_prefix"].removeprefix("s3://")
            bucket, prefix = destination.split("/", 1)
            prefix = prefix.rstrip("/") + "/"
            boundary_residual = (
                1
                if int(row["partial_object_count"]) > 0
                and int(row["allowed_materialized_target_residual_uint32_values"]) >= 1
                else 0
            )
            planned_values = int(row["planned_uint32_values"]) + boundary_residual
            shard_count = int(row["planned_output_shard_count"])
            base_values, extra_values = divmod(planned_values, shard_count)
            output_by_prefix[(bucket, prefix)] = [
                obj
                for shard_index in range(shard_count)
                for obj in (
                    S3Object(
                        bucket,
                        prefix + f"{shard_index:06d}.npy",
                        (base_values + (shard_index < extra_values)) * 4,
                        "output-etag",
                    ),
                    S3Object(
                        bucket,
                        prefix + f"{shard_index:06d}.csv.gz",
                        24,
                        "metadata-etag",
                    ),
                )
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
        output_summary = json.loads((self.build / "03-output-validation/output-summary.json").read_text())
        self.assertTrue(output_summary["aggregate_target_residual_within_bound"])
        self.assertEqual(
            output_summary["actual_output_shard_count"],
            output_summary["planned_output_shard_count"],
        )
        self.assertNotEqual(
            output_summary["actual_uint32_values"],
            output_summary["predicted_uint32_values"],
        )
        propose_configs(
            argparse.Namespace(
                build=self.build,
                destination_root="s3://test-bucket/new-datasets/dolma3p5",
                local_temp_root=str(self.root / "temp-base"),
                max_unit_working_bytes=20_000_000_000_000,
            )
        )
        self.assertTrue((self.build / "01-plan/execution/proposal-summary.json").is_file())
        self.assertFalse((self.build / "02-preflight").exists())
        self.assertFalse((self.build / "03-output-validation").exists())

    def test_plan_refuses_to_replace_unknown_files_in_a_build(self):
        self._plan()
        critical_file = self.build / "source-tokens.npy"
        critical_file.write_bytes(b"do not replace")

        with self.assertRaisesRegex(PreparationError, "unknown top-level entries"):
            self._plan()

        self.assertEqual(critical_file.read_bytes(), b"do not replace")

    def test_allocation_is_deterministic_and_size_based(self):
        first = _allocate_object_sampling(26, [8, 12, 20])
        second = _allocate_object_sampling(26, [8, 12, 20])
        self.assertEqual(first, second)
        repetitions, partial_targets, planned = first
        self.assertEqual(
            planned,
            sum(
                size * repeat + partial
                for size, repeat, partial in zip(
                    [8, 12, 20], repetitions, partial_targets
                )
            ),
        )
        self.assertEqual(planned, 26)
        self.assertEqual(repetitions, [0, 0, 0])
        self.assertEqual(partial_targets, [5, 8, 13])

    def test_large_leaf_target_is_distributed_across_every_object(self):
        sizes = [2_993_314_462, 2_930_303_417, 2_852_281_877, 2_863_478_216]
        repetitions, partial_targets, planned = _allocate_object_sampling(
            4_810_877_400, sizes
        )
        self.assertEqual(planned, 4_810_877_400)
        self.assertEqual(repetitions, [0, 0, 0, 0])
        self.assertTrue(all(value > 0 for value in partial_targets))
        self.assertEqual(sum(partial_targets), planned)

    def test_fractional_rate_samples_the_same_fraction_from_every_shard(self):
        sizes = [1_000, 2_000, 3_000, 4_000]
        downsample_repeats, downsample_partials, downsample_total = (
            _allocate_object_sampling(3_000, sizes)
        )
        self.assertEqual(downsample_repeats, [0, 0, 0, 0])
        self.assertEqual(downsample_partials, [300, 600, 900, 1_200])
        self.assertEqual(downsample_total, 3_000)

        upsample_repeats, upsample_partials, upsample_total = _allocate_object_sampling(
            13_000, sizes
        )
        self.assertEqual(upsample_repeats, [1, 1, 1, 1])
        self.assertEqual(upsample_partials, [300, 600, 900, 1_200])
        self.assertEqual(upsample_total, 13_000)

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

    def test_partial_manifest_selects_and_materializes_whole_documents(self):
        pair = self.root / "partial-pair"
        pair.mkdir()
        npy = pair / "tokens.npy"
        metadata = pair / "tokens.csv.gz"
        np.arange(100, dtype=np.uint32).tofile(npy)
        boundaries = [0, 7, 19, 44, 70, 100]
        with gzip.open(metadata, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                writer.writerow([start, end, f"doc-{index}", "source", index])

        manifest_path = self.root / "partial-manifest.csv"
        with manifest_path.open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "npy_uri",
                    "metadata_uri",
                    "repeat_count",
                    "partial_target_uint32_values",
                    "selection_seed",
                    "selection_algorithm",
                    "npy_size_bytes",
                    "metadata_size_bytes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "npy_uri": npy,
                    "metadata_uri": metadata,
                    "repeat_count": 0,
                    "partial_target_uint32_values": 43,
                    "selection_seed": 1234,
                    "selection_algorithm": "document_hash_bucket_v1",
                    "npy_size_bytes": npy.stat().st_size,
                    "metadata_size_bytes": metadata.stat().st_size,
                }
            )

        manifest = ReshardingManifestConfig(manifest_path)
        first = manifest.take(self.root / "partial-run-1", max_workers=1)
        second = manifest.take(self.root / "partial-run-2", max_workers=1)
        self.assertFalse((pair / "selection.csv.gz").exists())
        self.assertEqual(len(first), 1)
        self.assertEqual(
            first[0].selected_uint32_values,
            second[0].selected_uint32_values,
        )
        self.assertLessEqual(abs(first[0].selected_uint32_values - 43), 30)
        with gzip.open(first[0].selection_path, "rt", encoding="utf-8") as handle:
            first_selection = handle.read()
        with gzip.open(second[0].selection_path, "rt", encoding="utf-8") as handle:
            second_selection = handle.read()
        self.assertEqual(first_selection, second_selection)

        output = self.root / "partial-output/000000.npy"
        progress_updates = []
        merge_result = merge_group(
            first,
            output,
            np.dtype(np.uint32),
            progress=progress_updates.append,
        )
        materialized = np.fromfile(output, dtype=np.uint32)
        self.assertEqual(len(materialized), first[0].selected_uint32_values)
        previous_end = 0
        with gzip.open(output.with_suffix(".csv.gz"), "rt", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for row in rows:
            self.assertEqual(int(row[0]), previous_end)
            previous_end = int(row[1])
        self.assertEqual(previous_end, len(materialized))
        self.assertEqual(merge_result.document_count, len(rows))
        self.assertEqual(merge_result.token_copy_operations, len(rows))
        self.assertEqual(merge_result.output_bytes, output.stat().st_size)
        self.assertEqual(
            merge_result.metadata_bytes, output.with_suffix(".csv.gz").stat().st_size
        )
        self.assertTrue(progress_updates[-1].complete)
        self.assertEqual(
            progress_updates[-1].processed_values,
            merge_result.output_bytes // np.dtype(np.uint32).itemsize,
        )
        with self.assertRaises(FileExistsError):
            merge_group(first, output, np.dtype(np.uint32))

    def test_partial_manifest_samples_documents_from_every_source_shard(self):
        manifest_rows = []
        for shard_index, seed in enumerate((101, 202, 303)):
            pair = self.root / f"distributed-pair-{shard_index}"
            pair.mkdir()
            npy = pair / "tokens.npy"
            metadata = pair / "tokens.csv.gz"
            np.arange(
                shard_index * 100,
                (shard_index + 1) * 100,
                dtype=np.uint32,
            ).tofile(npy)
            with gzip.open(metadata, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                for document_index in range(10):
                    start = document_index * 10
                    writer.writerow(
                        [
                            start,
                            start + 10,
                            f"shard-{shard_index}-doc-{document_index}",
                            f"source-{shard_index}",
                            document_index,
                        ]
                    )
            manifest_rows.append(
                {
                    "npy_uri": npy,
                    "metadata_uri": metadata,
                    "repeat_count": 0,
                    "partial_target_uint32_values": 30,
                    "selection_seed": seed,
                    "selection_algorithm": "document_hash_bucket_v1",
                    "npy_size_bytes": npy.stat().st_size,
                    "metadata_size_bytes": metadata.stat().st_size,
                }
            )

        manifest_path = self.root / "distributed-partial-manifest.csv"
        with manifest_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

        paths = ReshardingManifestConfig(manifest_path).take(
            self.root / "distributed-partial-run", max_workers=2
        )
        self.assertEqual(len(paths), 3)
        selected_index_sets = []
        for shard_index, path in enumerate(paths):
            self.assertEqual(path.selected_uint32_values, 30)
            with gzip.open(path.selection_path, "rt", encoding="utf-8") as handle:
                selected_rows = list(csv.reader(handle))
            self.assertEqual(len(selected_rows), 3)
            self.assertTrue(
                all(row[2].startswith(f"shard-{shard_index}-") for row in selected_rows)
            )
            selected_index_sets.append(frozenset(int(row[4]) for row in selected_rows))
        self.assertGreater(len(set(selected_index_sets)), 1)

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

    def test_remote_manifest_streams_s5cmd_with_large_file_concurrency(self):
        manifest_path = self.root / "remote-manifest.csv"
        with manifest_path.open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "npy_uri",
                    "metadata_uri",
                    "repeat_count",
                    "npy_size_bytes",
                    "metadata_size_bytes",
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
                }
            )

        client = MagicMock()
        client.head_object.side_effect = lambda **request: {
            "ContentLength": 16 if request["Key"].endswith(".npy") else 12,
            "ETag": '"source"',
        }

        def emulate_s5cmd(command: list[str], phase: str) -> None:
            self.assertEqual(
                command[:5], ["s5cmd", "--stat", "--numworkers", "3", "run"]
            )
            self.assertEqual(phase, "source download")
            command_lines = Path(command[-1]).read_text().splitlines()
            self.assertEqual(len(command_lines), 2)
            for command_line, expected_size in zip(command_lines, (16, 12)):
                arguments = shlex.split(command_line)
                self.assertEqual(
                    arguments[:6],
                    [
                        "cp",
                        "--show-progress",
                        "--raw",
                        "--no-clobber",
                        "--concurrency",
                        "17",
                    ],
                )
                destination = Path(arguments[-1])
                destination.write_bytes(b"\x00" * expected_size)

        with (
            patch("dolma.tokenizer.reshard.boto3.client", return_value=client),
            patch("dolma.tokenizer.reshard._run_s5cmd", side_effect=emulate_s5cmd),
        ):
            paths = ReshardingManifestConfig(manifest_path).take(
                self.root / "remote-input",
                max_workers=3,
                s5cmd_concurrency=17,
            )

        self.assertEqual(len(paths), 1)
        self.assertEqual(Path(paths[0].npy_path).stat().st_size, 16)
        self.assertEqual(Path(paths[0].csv_path).stat().st_size, 12)

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
        with patch("dolma.tokenizer.reshard._run_s5cmd") as run:
            upload_to_s3(self.root, "s3://test-bucket/new/prefix", max_workers=3)
        command = run.call_args.args[0]
        self.assertIn("--no-clobber", command)
        self.assertIn("--show-progress", command)
        self.assertEqual(command[:5], ["s5cmd", "--stat", "--numworkers", "3", "cp"])

    def test_s3_bucket_root_is_refused_before_listing(self):
        with self.assertRaises(ValueError):
            destination_has_objects("s3://test-bucket")
        client = MagicMock()
        client.list_objects_v2.return_value = {"Contents": [{"Key": "existing"}]}
        with patch("dolma.tokenizer.reshard.boto3.client", return_value=client):
            self.assertTrue(destination_has_objects("s3://test-bucket/new/prefix"))

    def test_worker_storage_setup_is_valid_and_defaults_to_no_action(self):
        syntax = subprocess.run(
            ["bash", "-n", str(WORKER_STORAGE_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        help_result = subprocess.run(
            ["bash", str(WORKER_STORAGE_SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--check", help_result.stdout)
        self.assertIn("--apply", help_result.stdout)

        no_mode = subprocess.run(
            ["bash", str(WORKER_STORAGE_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(no_mode.returncode, 2)

    def test_worker_storage_setup_only_selects_instance_store_devices(self):
        script = WORKER_STORAGE_SCRIPT.read_text()
        self.assertIn("EC2 NVMe Instance Storage", script)
        self.assertIn('sudo wipefs -n "$device"', script)
        self.assertIn('if [[ "$mode" == "--check" ]]', script)
        self.assertNotIn("/dev/nvme1n1", script)

    def test_worker_storage_check_plans_direct_mount_or_raid0(self):
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        fake_lsblk = fake_bin / "lsblk"
        fake_lsblk.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "-dpno NAME,TYPE" ]]; then
  echo "/dev/nvme0n1 disk"
  for ((i = 1; i <= FAKE_NVME_COUNT; i++)); do
    echo "/dev/nvme${i}n1 disk"
  done
elif [[ "$1" == "-dno" && "$2" == "MODEL" ]]; then
  if [[ "$3" == "/dev/nvme0n1" ]]; then
    echo "Amazon Elastic Block Store"
  else
    echo "Amazon EC2 NVMe Instance Storage"
  fi
elif [[ "$1" == "-bdno" && "$2" == "SIZE" ]]; then
  echo "$FAKE_NVME_SIZE_BYTES"
elif [[ "$1" == "-nrpo" && "$2" == "NAME,TYPE" ]]; then
  echo "$3 disk"
elif [[ "$1" == "-nrpo" && "$2" == "MOUNTPOINT" ]]; then
  echo
else
  echo "unexpected lsblk arguments: $*" >&2
  exit 97
fi
"""
        )
        fake_sudo = fake_bin / "sudo"
        fake_sudo.write_text(
            """#!/usr/bin/env bash
if [[ "$1" == "wipefs" && "$2" == "-n" ]]; then
  exit 0
fi
echo "unexpected sudo arguments: $*" >&2
exit 97
"""
        )
        fake_findmnt = fake_bin / "findmnt"
        fake_findmnt.write_text("#!/usr/bin/env bash\nexit 1\n")
        for command in (fake_lsblk, fake_sudo, fake_findmnt):
            command.chmod(0o755)

        environment = os.environ.copy()
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
        for device_count, device_size, layout, expected in (
            (1, 1_875_000_000_000, "auto", "Plan: format /dev/nvme1n1 as XFS"),
            (2, 1_875_000_000_000, "auto", "Layout decision required"),
            (2, 1_875_000_000_000, "single", "Plan: format /dev/nvme1n1 as XFS"),
            (2, 1_875_000_000_000, "raid0", "create RAID0 across 2 devices"),
        ):
            environment["FAKE_NVME_COUNT"] = str(device_count)
            environment["FAKE_NVME_SIZE_BYTES"] = str(device_size)
            result = subprocess.run(
                [
                    "bash",
                    str(WORKER_STORAGE_SCRIPT),
                    "--check",
                    "--layout",
                    layout,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                f"Detected {device_count} EC2 instance-store device(s)",
                result.stdout,
            )
            self.assertIn(expected, result.stdout)
            self.assertIn("no storage changes were made", result.stdout)

        environment["FAKE_NVME_COUNT"] = "2"
        environment["FAKE_NVME_SIZE_BYTES"] = "1875000000000"
        implicit_apply = subprocess.run(
            ["bash", str(WORKER_STORAGE_SCRIPT), "--apply"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(implicit_apply.returncode, 1)
        self.assertIn("Refusing to apply an implicit layout", implicit_apply.stderr)


if __name__ == "__main__":
    unittest.main()
