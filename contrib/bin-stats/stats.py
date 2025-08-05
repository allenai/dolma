#!/usr/bin/env python3
"""
Analyze bucketed JSONL.zst files and generate distribution plots and statistics.
"""

import argparse
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import msgspec


class Metadata(msgspec.Struct):
    """Struct for extracting only the score field from metadata."""
    score: Optional[float] = None


class Record(msgspec.Struct):
    """Struct for extracting only the metadata field from records."""
    metadata: Optional[Metadata] = None


def count_lines_in_zst_file(file_path: Path) -> int:
    """Count lines in a zstandard compressed JSONL file."""
    try:
        with open(file_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            try:
                with dctx.stream_reader(fh) as reader:
                    line_count = 0
                    buffer = b""
                    while True:
                        chunk = reader.read(8192)
                        if not chunk:
                            break
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if line.strip():
                                line_count += 1
                    # Handle last line if no trailing newline
                    if buffer.strip():
                        line_count += 1
                    return line_count
            except zstd.ZstdError as e:
                print(f"Zstandard decompression error for {file_path}: {e}")
                return 0
    except (OSError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        return 0


def process_zst_file_threaded(file_path: Path, num_threads: int = 4) -> Tuple[int, List[float]]:
    """Process a single zst file using multiple threads for line processing."""
    line_count = 0
    scores = []
    
    # Read entire file into memory first to utilize high memory
    try:
        with open(file_path, "rb") as fh:
            compressed_data = fh.read()
        
        # Decompress all data at once to utilize memory
        dctx = zstd.ZstdDecompressor()
        try:
            decompressed_data = dctx.decompress(compressed_data)
        except zstd.ZstdError as e:
            print(f"Zstandard decompression error for {file_path}: {e}")
            return 0, []
        
        # Split into lines
        lines = decompressed_data.split(b"\n")
        lines = [line for line in lines if line.strip()]
        
        # Process lines in parallel using threadpool
        def process_line(line_data):
            try:
                record = msgspec.json.decode(line_data, type=Record)
                score = None
                if record.metadata and record.metadata.score is not None:
                    score = record.metadata.score
                return 1, score
            except (msgspec.DecodeError, ValueError, TypeError):
                return 1, None
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_line, lines))
        
        # Aggregate results
        for count, score in results:
            line_count += count
            if score is not None:
                scores.append(score)
                
    except (OSError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return 0, []
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return 0, []

    return line_count, scores


def extract_scores_from_zst_file(file_path: Path) -> List[float]:
    """Extract metadata.score values from a zstandard compressed JSONL file."""
    scores = []
    try:
        with open(file_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            try:
                with dctx.stream_reader(fh) as reader:
                    buffer = b""
                    while True:
                        chunk = reader.read(8192)
                        if not chunk:
                            break
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if line.strip():
                                try:
                                    record = msgspec.json.decode(line, type=Record)
                                    if record.metadata and record.metadata.score is not None:
                                        scores.append(record.metadata.score)
                                except (msgspec.DecodeError, ValueError, TypeError):
                                    continue
                    # Handle last line if no trailing newline
                    if buffer.strip():
                        try:
                            record = msgspec.json.decode(buffer, type=Record)
                            if record.metadata and record.metadata.score is not None:
                                scores.append(record.metadata.score)
                        except (msgspec.DecodeError, ValueError, TypeError):
                            pass
            except zstd.ZstdError as e:
                print(f"Zstandard decompression error for {file_path}: {e}")
                return []
    except (OSError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error extracting scores from {file_path}: {e}")
    return scores


def process_bucket_sequential(bucket_info: Tuple[Path, str, 'queue.Queue', int]) -> Tuple[str, Dict[str, Any]]:
    """Process a single bucket with sequential file processing but threaded line processing."""
    bucket_dir, bucket_name, progress_queue, threads_per_file = bucket_info
    
    # Process all .jsonl.zst files in the bucket (including subdirectories)
    zst_files = list(bucket_dir.glob("**/*.jsonl.zst"))
    if not zst_files:
        return bucket_name, {
            "total_lines": 0,
            "file_count": 0,
            "scores": [],
            "score_min": None,
            "score_max": None,
            "score_median": None,
            "score_mean": None,
            "score_std": None,
            "score_q25": None,
            "score_q75": None,
        }

    file_count = len(zst_files)
    total_lines = 0
    all_scores = []
    
    # Process files sequentially within each bucket
    for file_path in zst_files:
        try:
            lines, scores = process_zst_file_threaded(file_path, threads_per_file)
            total_lines += lines
            all_scores.extend(scores)
            
            if progress_queue:
                progress_queue.put(('file_completed', bucket_name, file_path.name))
        except Exception as e:
            print(f"Error processing {file_path} in bucket {bucket_name}: {e}")
            if progress_queue:
                progress_queue.put(('file_completed', bucket_name, file_path.name))

    # Create bucket statistics
    base_stats = {
        "total_lines": total_lines,
        "file_count": file_count,
        "scores": all_scores,
    }

    if all_scores:
        score_array = np.array(all_scores)
        base_stats.update(
            {
                "score_min": float(np.min(score_array)),
                "score_max": float(np.max(score_array)),
                "score_median": float(np.median(score_array)),
                "score_mean": float(np.mean(score_array)),
                "score_std": float(np.std(score_array)),
                "score_q25": float(np.percentile(score_array, 25)),
                "score_q75": float(np.percentile(score_array, 75)),
            }
        )
    else:
        base_stats.update(
            {
                "score_min": None,
                "score_max": None,
                "score_median": None,
                "score_mean": None,
                "score_std": None,
                "score_q25": None,
                "score_q75": None,
            }
        )

    return bucket_name, base_stats


def analyze_buckets(input_dir: Path, num_workers: int = 128) -> Dict[str, Dict[str, Any]]:
    """Analyze all buckets using concurrent bucket processing with sequential file processing per bucket."""
    bucket_stats = {}

    # Scan all buckets first to get total count and file counts
    bucket_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    # Count total files for accurate progress reporting
    print("Scanning files for progress tracking...")
    total_files = 0
    bucket_file_counts = {}
    for bucket_dir in bucket_dirs:
        file_count = len(list(bucket_dir.glob("**/*.jsonl.zst")))
        bucket_file_counts[bucket_dir.name] = file_count
        total_files += file_count
    
    print(f"Found {len(bucket_dirs)} buckets with {total_files:,} total files")
    
    # Configure parallelization: concurrent buckets, sequential files within bucket, threaded line processing
    total_cores = cpu_count()  # Should be 128
    num_bucket_workers = min(len(bucket_dirs), max(1, total_cores // 8))  # Allow more bucket workers
    threads_per_file = max(1, total_cores // num_bucket_workers) if num_bucket_workers > 0 else 4
    print(f"Using {num_bucket_workers} concurrent bucket processes")
    print(f"Each file processed with {threads_per_file} threads for line processing")
    print(f"Files within each bucket processed sequentially to utilize high memory")
    
    # Create progress tracking
    console = Console()
    progress_queue = Manager().Queue()
    
    def progress_monitor(progress_queue, total_files, bucket_file_counts):
        """Monitor and update progress based on completed files."""
        completed_files = 0
        bucket_progress = {name: 0 for name in bucket_file_counts.keys()}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total} files)"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            file_task = progress.add_task("Processing files", total=total_files)
            
            while completed_files < total_files:
                try:
                    msg = progress_queue.get(timeout=1.0)
                    if msg[0] == 'file_completed':
                        _, bucket_name, file_name = msg
                        completed_files += 1
                        bucket_progress[bucket_name] += 1
                        
                        # Update progress with global file count only
                        progress.update(file_task, 
                                      description="Processing files",
                                      completed=completed_files)
                except queue.Empty:
                    continue
                except:
                    break
    
    # Start progress monitoring in background thread
    bucket_info_list = [(bucket_dir, bucket_dir.name, progress_queue, threads_per_file) for bucket_dir in bucket_dirs]
    
    progress_thread = threading.Thread(
        target=progress_monitor, 
        args=(progress_queue, total_files, bucket_file_counts)
    )
    progress_thread.daemon = True
    progress_thread.start()
    
    # Process buckets in parallel
    if num_bucket_workers > 1:
        with Pool(processes=num_bucket_workers) as pool:
            results = pool.map(process_bucket_sequential, bucket_info_list)
            for bucket_name, stats in results:
                bucket_stats[bucket_name] = stats
    else:
        # Process buckets sequentially
        for bucket_info in bucket_info_list:
            bucket_name, stats = process_bucket_sequential(bucket_info)
            bucket_stats[bucket_name] = stats
    
    # Signal completion and wait for progress thread
    for _ in range(total_files):
        try:
            progress_queue.put(('file_completed', 'DONE', 'DONE'))
        except:
            break
    
    time.sleep(0.5)  # Give progress thread time to finish
    
    return bucket_stats


def create_visualizations(bucket_stats: Dict[str, Dict[str, Any]]) -> None:
    """Create distribution plots and visualizations."""
    # Prepare data for plotting
    bucket_names = list(bucket_stats.keys())
    line_counts = [stats["total_lines"] for stats in bucket_stats.values()]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Bucket Analysis", fontsize=16)

    # 1. Distribution of line counts across buckets
    ax1 = axes[0, 0]
    sns.barplot(x=bucket_names, y=line_counts, ax=ax1)
    ax1.set_title("Lines per Bucket")
    ax1.set_xlabel("Bucket")
    ax1.set_ylabel("Number of Lines")
    ax1.tick_params(axis="x", rotation=45)

    # Format y-axis with comma separators
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    # 2. Score distribution across all buckets
    ax2 = axes[0, 1]
    all_scores = [score for stats in bucket_stats.values() if stats["scores"] for score in stats["scores"]]

    if all_scores:
        sns.histplot(all_scores, bins=50, ax=ax2)
        ax2.set_title("Score Distribution (All Buckets)")
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Frequency")

    # 3. Box plot of scores by bucket
    ax3 = axes[1, 0]
    buckets_with_scores = [
        (bucket_name, stats["scores"]) for bucket_name, stats in bucket_stats.items() if stats["scores"]
    ]

    if buckets_with_scores:
        bucket_labels, score_data = zip(*buckets_with_scores)
        bucket_labels = list(bucket_labels)
        score_data = list(score_data)

    if buckets_with_scores:
        # Handle buckets with different numbers of scores
        max_len = max(len(scores) for scores in score_data)
        padded_data = {}
        for label, scores in zip(bucket_labels, score_data):
            padded_data[label] = scores + [np.nan] * (max_len - len(scores))

        df_scores = pd.DataFrame(padded_data)
        df_melted = df_scores.melt(var_name="Bucket", value_name="Score")
        df_melted = df_melted.dropna(subset=["Score"])
        sns.boxplot(data=df_melted, x="Bucket", y="Score", ax=ax3)
        ax3.set_title("Score Distribution by Bucket")
        ax3.tick_params(axis="x", rotation=45)

    # 4. Scatter plot of median score vs line count
    ax4 = axes[1, 1]
    buckets_with_median = [
        (bucket_name, stats["score_median"], stats["total_lines"])
        for bucket_name, stats in bucket_stats.items()
        if stats["score_median"] is not None
    ]

    if buckets_with_median:
        bucket_names_with_scores, median_scores, line_counts_with_scores = zip(*buckets_with_median)
        bucket_names_with_scores = list(bucket_names_with_scores)
        median_scores = list(median_scores)
        line_counts_with_scores = list(line_counts_with_scores)

    if buckets_with_median:
        ax4.scatter(line_counts_with_scores, median_scores, alpha=0.7)
        ax4.set_title("Median Score vs Line Count")
        ax4.set_xlabel("Lines in Bucket")
        ax4.set_ylabel("Median Score")
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

        # Add bucket labels to points
        for bucket_name, median_score, line_count in zip(
            bucket_names_with_scores, median_scores, line_counts_with_scores
        ):
            ax4.annotate(
                bucket_name, (line_count, median_score), xytext=(5, 5), textcoords="offset points", fontsize=8
            )

    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display plots: {e}")
        print("Plots may not be available in headless environments")


def print_statistics(bucket_stats: Dict[str, Dict[str, Any]]) -> None:
    """Print detailed statistics about the buckets."""
    print("\n" + "=" * 80)
    print("BUCKET ANALYSIS SUMMARY")
    print("=" * 80)

    # Overall statistics
    total_lines = sum(stats["total_lines"] for stats in bucket_stats.values())
    total_files = sum(stats["file_count"] for stats in bucket_stats.values())
    total_buckets = len(bucket_stats)

    print(f"Total buckets: {total_buckets}")
    print(f"Total files: {total_files:,}")
    print(f"Total lines: {total_lines:,}")
    if total_buckets > 0:
        print(f"Average lines per bucket: {total_lines / total_buckets:,.0f}")
    else:
        print("Average lines per bucket: 0")

    # Sort buckets by line count
    sorted_buckets = sorted(bucket_stats.items(), key=lambda x: x[1]["total_lines"], reverse=True)

    print("\nBUCKET DETAILS (sorted by line count):")
    print("-" * 80)
    print(f"{'Bucket':<15} {'Lines':<12} {'Files':<8} {'Score Stats':<30}")
    print("-" * 80)

    for bucket_name, stats in sorted_buckets:
        score_info = ""
        if stats["score_median"] is not None:
            score_info = (
                f"min:{stats['score_min']:.3f} med:{stats['score_median']:.3f} max:{stats['score_max']:.3f}"
            )
        else:
            score_info = "No scores found"

        print(f"{bucket_name:<15} {stats['total_lines']:<12,} {stats['file_count']:<8} {score_info:<30}")

    # Score analysis
    print(f"\nSCORE ANALYSIS:")
    print("-" * 80)

    buckets_with_scores = [
        (name, stats) for name, stats in bucket_stats.items() if stats["score_median"] is not None
    ]

    if buckets_with_scores:
        print(f"{'Bucket':<15} {'Count':<10} {'Mean':<8} {'Std':<8} {'Q25':<8} {'Q75':<8}")
        print("-" * 80)

        for bucket_name, stats in sorted(buckets_with_scores, key=lambda x: x[1]["score_median"], reverse=True):
            print(
                f"{bucket_name:<15} {len(stats['scores']):<10,} "
                f"{stats['score_mean']:<8.3f} {stats['score_std']:<8.3f} "
                f"{stats['score_q25']:<8.3f} {stats['score_q75']:<8.3f}"
            )

        # Correlation analysis
        if len(buckets_with_scores) > 1:
            line_counts = [stats["total_lines"] for _, stats in buckets_with_scores]
            median_scores = [stats["score_median"] for _, stats in buckets_with_scores]
            try:
                correlation_matrix = np.corrcoef(line_counts, median_scores)
                correlation = correlation_matrix[0, 1]
                if not np.isnan(correlation):
                    print(f"\nCorrelation between bucket size and median score: {correlation:.3f}")
                else:
                    print("\nCorrelation could not be computed (insufficient variance)")
            except (ValueError, IndexError):
                print("\nCorrelation could not be computed")
    else:
        print("No score data found in any bucket.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze bucketed JSONL.zst files")
    parser.add_argument("input_dir", type=Path, help="Input directory containing bucket subdirectories")
    parser.add_argument("--workers", type=int, default=128, help="Number of total cores to use (default: 128)")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        return 1

    print(f"Analyzing buckets in: {args.input_dir}")

    # Analyze buckets
    bucket_stats = analyze_buckets(args.input_dir, args.workers)

    if not bucket_stats:
        print("No buckets found in the input directory")
        return 1

    # Print statistics
    print_statistics(bucket_stats)

    # Create visualizations
    create_visualizations(bucket_stats)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
