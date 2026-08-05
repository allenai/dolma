"""Deterministic, document-boundary sampling for token memmaps.

The token count used by the planner is still derived from the uint32 memmap
size.  This module reads the paired metadata only while materializing a partial
copy, and writes a compact metadata index describing the selected documents.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import smart_open

DOCUMENT_SELECTION_ALGORITHM = "document_hash_bucket_v1"
DOCUMENT_HASH_BUCKETS = 1 << 16


@dataclass(frozen=True)
class DocumentMetadataRow:
    row_number: int
    start: int
    end: int
    document_id: str
    source: str
    source_index: int

    @property
    def token_count(self) -> int:
        return self.end - self.start

    def as_csv_row(self) -> list[str | int]:
        return [
            self.start,
            self.end,
            self.document_id,
            self.source,
            self.source_index,
        ]


@dataclass(frozen=True)
class DocumentSelectionResult:
    selection_path: Path
    requested_uint32_values: int
    selected_uint32_values: int
    target_residual_uint32_values: int
    selected_document_count: int
    source_document_count: int
    largest_document_uint32_values: int


def _iter_metadata_rows(
    metadata_path: Path, source_uint32_values: int
) -> Iterator[DocumentMetadataRow]:
    previous_end = 0
    with smart_open.open(metadata_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 5:
                raise ValueError(
                    f"Invalid token metadata row {metadata_path}:{row_number}; "
                    "expected five columns"
                )
            try:
                start = int(row[0])
                end = int(row[1])
                source_index = int(row[4])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid token offsets on {metadata_path}:{row_number}"
                ) from exc
            if start != previous_end or end <= start or end > source_uint32_values:
                raise ValueError(
                    f"Non-contiguous token metadata on {metadata_path}:{row_number}; "
                    f"expected start {previous_end}, found [{start}, {end})"
                )
            previous_end = end
            yield DocumentMetadataRow(
                row_number=row_number,
                start=start,
                end=end,
                document_id=row[2],
                source=row[3],
                source_index=source_index,
            )
    if previous_end != source_uint32_values:
        raise ValueError(
            f"Token metadata does not cover its memmap: {metadata_path}; "
            f"covered {previous_end} of {source_uint32_values} uint32 values"
        )


def _document_hash(row: DocumentMetadataRow, seed: int) -> int:
    payload = (
        f"{seed}\0{row.start}\0{row.end}\0{row.document_id}\0"
        f"{row.source}\0{row.source_index}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(
            payload,
            digest_size=8,
            person=b"dolma-doc-v1",
        ).digest(),
        "big",
    )


def _hash_bucket(hash_value: int) -> int:
    return hash_value >> (64 - (DOCUMENT_HASH_BUCKETS.bit_length() - 1))


def create_document_selection(
    *,
    metadata_path: str | Path,
    selection_path: str | Path,
    source_uint32_values: int,
    target_uint32_values: int,
    seed: int,
    progress: Callable[[str, int, int], None] | None = None,
    progress_interval_seconds: float = 10.0,
) -> DocumentSelectionResult:
    """Select a deterministic hash-ranked prefix of whole documents.

    A first pass totals document lengths by hash bucket. A second pass streams
    rows below the threshold into the selection index and retains only the
    threshold bucket in memory. The realized count differs from the requested
    count by no more than the largest document in the source object.
    """

    metadata_path = Path(metadata_path)
    selection_path = Path(selection_path)
    if source_uint32_values <= 0:
        raise ValueError("source_uint32_values must be positive")
    if not 0 < target_uint32_values < source_uint32_values:
        raise ValueError(
            "target_uint32_values must be between zero and the source size"
        )
    if selection_path.exists() or selection_path.is_symlink():
        raise FileExistsError(f"Refusing to replace selection index: {selection_path}")

    bucket_values = [0] * DOCUMENT_HASH_BUCKETS
    source_document_count = 0
    largest_document = 0
    last_progress_at = time.monotonic()
    if progress is not None:
        progress("pass 1/2", 0, 0)
    for row in _iter_metadata_rows(metadata_path, source_uint32_values):
        bucket_values[_hash_bucket(_document_hash(row, seed))] += row.token_count
        source_document_count += 1
        largest_document = max(largest_document, row.token_count)
        if (
            progress is not None
            and time.monotonic() - last_progress_at >= progress_interval_seconds
        ):
            progress("pass 1/2", source_document_count, row.end)
            last_progress_at = time.monotonic()
    if progress is not None:
        progress("pass 1/2", source_document_count, source_uint32_values)

    below_threshold_values = 0
    threshold_bucket = -1
    threshold_target = 0
    for bucket, bucket_total in enumerate(bucket_values):
        if below_threshold_values + bucket_total < target_uint32_values:
            below_threshold_values += bucket_total
            continue
        threshold_bucket = bucket
        threshold_target = target_uint32_values - below_threshold_values
        break
    if threshold_bucket < 0:
        raise RuntimeError("Could not locate the document-selection hash threshold")

    selected_documents = 0
    threshold_rows: list[tuple[int, DocumentMetadataRow]] = []
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    second_pass_documents = 0
    last_progress_at = time.monotonic()
    if progress is not None:
        progress("pass 2/2", 0, 0)
    with gzip.open(selection_path, "xt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in _iter_metadata_rows(metadata_path, source_uint32_values):
            second_pass_documents += 1
            hash_value = _document_hash(row, seed)
            bucket = _hash_bucket(hash_value)
            if bucket < threshold_bucket:
                writer.writerow(row.as_csv_row())
                selected_documents += 1
            elif bucket == threshold_bucket:
                threshold_rows.append((hash_value, row))
            if (
                progress is not None
                and time.monotonic() - last_progress_at >= progress_interval_seconds
            ):
                progress("pass 2/2", second_pass_documents, row.end)
                last_progress_at = time.monotonic()

        if progress is not None:
            progress("pass 2/2", second_pass_documents, source_uint32_values)

        threshold_rows.sort(
            key=lambda item: (
                item[0],
                item[1].start,
                item[1].end,
                item[1].row_number,
            )
        )
        selected_threshold_values = 0
        best_error = threshold_target
        for _, row in threshold_rows:
            candidate_values = selected_threshold_values + row.token_count
            candidate_error = abs(threshold_target - candidate_values)
            if candidate_error > best_error:
                break
            writer.writerow(row.as_csv_row())
            selected_documents += 1
            selected_threshold_values = candidate_values
            best_error = candidate_error

    selected_values = below_threshold_values + selected_threshold_values

    residual = selected_values - target_uint32_values
    if abs(residual) > largest_document:
        raise RuntimeError(
            "Document selection exceeded its one-document residual bound: "
            f"{residual} versus largest document {largest_document}"
        )
    return DocumentSelectionResult(
        selection_path=selection_path,
        requested_uint32_values=target_uint32_values,
        selected_uint32_values=selected_values,
        target_residual_uint32_values=residual,
        selected_document_count=selected_documents,
        source_document_count=source_document_count,
        largest_document_uint32_values=largest_document,
    )
