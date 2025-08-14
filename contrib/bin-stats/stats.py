#!/usr/bin/env python3
"""
Analyze bucketed JSONL.zst files and generate distribution plots and statistics.
"""

import argparse
import io
import json
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
        
        # Decompress all data using streaming approach to handle frames without content size
        dctx = zstd.ZstdDecompressor()
        try:
            # Use streaming decompression to handle frames without content size in header
            with dctx.stream_reader(io.BytesIO(compressed_data)) as reader:
                decompressed_data = reader.read()
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
            "scores_by_language": {},
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
    scores_by_language = {}
    
    # Process files sequentially within each bucket
    for file_path in zst_files:
        try:
            lines, scores = process_zst_file_threaded(file_path, threads_per_file)
            total_lines += lines
            all_scores.extend(scores)
            
            # Extract language from directory structure (e.g., bucket/C/part_0000.jsonl.zst)
            language = file_path.parent.name
            if language not in scores_by_language:
                scores_by_language[language] = []
            scores_by_language[language].extend(scores)
            
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
        "scores_by_language": scores_by_language,
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


def sort_bucket_names(bucket_names: List[str]) -> List[str]:
    """Sort bucket names correctly, handling length_2e9, length_2e10, etc."""
    def bucket_sort_key(name: str) -> float:
        # Extract numeric part from bucket names like "length_2e9", "length_2e10", etc.
        if name.startswith("length_"):
            try:
                # Handle scientific notation like "2e9", "2e10"
                numeric_part = name.replace("length_", "")
                return float(numeric_part)
            except ValueError:
                pass
        # Fallback to string sorting for non-standard names
        return float('inf')
    
    return sorted(bucket_names, key=bucket_sort_key)


def create_visualizations(bucket_stats: Dict[str, Dict[str, Any]]) -> None:
    """Create distribution plots and visualizations using seaborn."""
    # Prepare data for plotting with proper sorting
    sorted_bucket_names = sort_bucket_names(list(bucket_stats.keys()))
    document_counts = [bucket_stats[name]["total_lines"] for name in sorted_bucket_names]
    all_scores = [score for stats in bucket_stats.values() if stats["scores"] for score in stats["scores"]]

    # Set seaborn style for better-looking plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create figure with subplots - 2x2 grid with 4 plots total
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # 1. Bar plot of lines per bucket (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(x=sorted_bucket_names, y=document_counts, ax=ax1)
    ax1.set_title("Documents per Bucket", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Bucket Name", fontsize=12)
    ax1.set_ylabel("Number of Documents", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    # Improve x-axis labels - rotate and show every nth label
    ax1.tick_params(axis="x", rotation=45, labelsize=10)
    step = max(1, len(sorted_bucket_names)//10)
    ticks = range(0, len(sorted_bucket_names), step)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([sorted_bucket_names[i] for i in ticks])

    # 2. Score distribution across all buckets with KDE (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    if all_scores:
        sns.histplot(all_scores, kde=True, bins=50, ax=ax2)
        ax2.set_title("Score Distribution (All Buckets)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Score", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)

    # 3. Violin plot of scores by bucket (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    buckets_with_scores = [
        (bucket_name, stats["scores"]) for bucket_name, stats in bucket_stats.items() if stats["scores"]
    ]
    
    if buckets_with_scores and len(buckets_with_scores) > 1:
        # Sort buckets for consistent ordering
        bucket_names_only = [bucket_name for bucket_name, _ in buckets_with_scores]
        sorted_bucket_names = sort_bucket_names(bucket_names_only)
        bucket_dict = {name: scores for name, scores in buckets_with_scores}
        
        # Create dataframe for violin plot with sorted buckets
        plot_data = []
        for bucket_name in sorted_bucket_names:
            if bucket_name in bucket_dict:
                scores = bucket_dict[bucket_name]
                # Limit to 500 scores per bucket and sample evenly for better performance and cleaner viz
                sample_size = min(500, len(scores))
                if len(scores) > sample_size:
                    # Sample evenly across the range
                    indices = np.linspace(0, len(scores) - 1, sample_size, dtype=int)
                    sampled_scores = [scores[i] for i in indices]
                else:
                    sampled_scores = scores
                for score in sampled_scores:
                    plot_data.append({'Bucket': bucket_name, 'Score': score})
        
        if plot_data:
            df_violin = pd.DataFrame(plot_data)
            # Use quartile inner marks instead of box for cleaner look
            sns.violinplot(data=df_violin, x="Bucket", y="Score", ax=ax3, order=sorted_bucket_names, inner="quartile", density_norm="width", cut=0)
            ax3.set_title("Score Distribution by Bucket", fontsize=14, fontweight='bold')
            ax3.set_xlabel("Bucket", fontsize=12)
            ax3.set_ylabel("Score", fontsize=12)
            # Improve x-axis label spacing - more aggressive filtering for violin plot
            ax3.tick_params(axis="x", rotation=45, labelsize=9)
            # Show only every 4th bucket to reduce clutter
            step = max(1, len(sorted_bucket_names) // 6)
            ticks = range(0, len(sorted_bucket_names), step)
            ax3.set_xticks(ticks)
            ax3.set_xticklabels([sorted_bucket_names[i] for i in ticks])

    # 4. Violin plot of scores by programming language (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Collect all scores by language across all buckets
    language_scores = {}
    for bucket_name, stats in bucket_stats.items():
        if "scores_by_language" in stats and stats["scores_by_language"]:
            for language, scores in stats["scores_by_language"].items():
                if language not in language_scores:
                    language_scores[language] = []
                language_scores[language].extend(scores)
    
    if language_scores:
        # Create dataframe for language violin plot
        language_plot_data = []
        sorted_languages = sorted(language_scores.keys())
        
        for language in sorted_languages:
            scores = language_scores[language]
            # Limit to 1000 scores per language for better performance
            sample_size = min(1000, len(scores))
            if len(scores) > sample_size:
                # Sample evenly across the range
                indices = np.linspace(0, len(scores) - 1, sample_size, dtype=int)
                sampled_scores = [scores[i] for i in indices]
            else:
                sampled_scores = scores
            for score in sampled_scores:
                language_plot_data.append({'Language': language, 'Score': score})
        
        if language_plot_data:
            df_language = pd.DataFrame(language_plot_data)
            sns.violinplot(data=df_language, x="Language", y="Score", ax=ax4, order=sorted_languages, inner="quartile", density_norm="width", cut=0)
            ax4.set_title("Score Distribution by Programming Language", fontsize=14, fontweight='bold')
            ax4.set_xlabel("Programming Language", fontsize=12)
            ax4.set_ylabel("Score", fontsize=12)
            ax4.tick_params(axis="x", rotation=45, labelsize=10)

    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.92, wspace=0.3, hspace=0.4)
    
    # Save the plot
    output_path = Path("bucket_analysis_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_path}")
    
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
        
        # Show programming language breakdown for this bucket
        if stats.get("scores_by_language"):
            language_stats = []
            for lang, lang_scores in stats["scores_by_language"].items():
                if lang_scores:
                    lang_array = np.array(lang_scores)
                    lang_stats = {
                        'count': len(lang_scores),
                        'mean': float(np.mean(lang_array)),
                        'median': float(np.median(lang_array)),
                        'min': float(np.min(lang_array)),
                        'max': float(np.max(lang_array))
                    }
                    language_stats.append((lang, lang_stats))
            
            if language_stats:
                # Sort languages by count descending
                language_stats.sort(key=lambda x: x[1]['count'], reverse=True)
                print(f"  {'Languages:':<14}")
                for lang, lang_stats in language_stats:
                    print(f"    {lang:<12} {lang_stats['count']:<8,} scores, "
                          f"mean:{lang_stats['mean']:.3f} med:{lang_stats['median']:.3f} "
                          f"[{lang_stats['min']:.3f}-{lang_stats['max']:.3f}]")

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

    # Programming Language Analysis
    print(f"\nPROGRAMMING LANGUAGE ANALYSIS:")
    print("-" * 80)
    
    # Collect all language stats across all buckets
    all_language_stats = {}
    for bucket_name, stats in bucket_stats.items():
        if stats.get("scores_by_language"):
            for lang, lang_scores in stats["scores_by_language"].items():
                if lang_scores:
                    if lang not in all_language_stats:
                        all_language_stats[lang] = []
                    all_language_stats[lang].extend(lang_scores)
    
    if all_language_stats:
        print(f"{'Language':<15} {'Count':<10} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8} {'Std':<8}")
        print("-" * 80)
        
        # Calculate stats for each language and sort by count
        language_summary = []
        for lang, all_scores in all_language_stats.items():
            lang_array = np.array(all_scores)
            lang_summary = {
                'count': len(all_scores),
                'mean': float(np.mean(lang_array)),
                'median': float(np.median(lang_array)),
                'min': float(np.min(lang_array)),
                'max': float(np.max(lang_array)),
                'std': float(np.std(lang_array))
            }
            language_summary.append((lang, lang_summary))
        
        # Sort by count descending
        language_summary.sort(key=lambda x: x[1]['count'], reverse=True)
        
        for lang, lang_stats in language_summary:
            print(f"{lang:<15} {lang_stats['count']:<10,} "
                  f"{lang_stats['mean']:<8.3f} {lang_stats['median']:<8.3f} "
                  f"{lang_stats['min']:<8.3f} {lang_stats['max']:<8.3f} "
                  f"{lang_stats['std']:<8.3f}")
    else:
        print("No programming language data found in any bucket.")


def save_results_to_json(bucket_stats: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """Save analysis results to JSON file for later reuse."""
    # Convert numpy types to native Python types for JSON serialization
    json_stats = {}
    for bucket_name, stats in bucket_stats.items():
        scores_by_language = {}
        if "scores_by_language" in stats and stats["scores_by_language"]:
            for lang, scores in stats["scores_by_language"].items():
                scores_by_language[lang] = [float(score) for score in scores]
        
        json_stats[bucket_name] = {
            "total_lines": int(stats["total_lines"]),
            "file_count": int(stats["file_count"]),
            "scores": [float(score) for score in stats["scores"]] if stats["scores"] else [],
            "scores_by_language": scores_by_language,
            "score_min": float(stats["score_min"]) if stats["score_min"] is not None else None,
            "score_max": float(stats["score_max"]) if stats["score_max"] is not None else None,
            "score_median": float(stats["score_median"]) if stats["score_median"] is not None else None,
            "score_mean": float(stats["score_mean"]) if stats["score_mean"] is not None else None,
            "score_std": float(stats["score_std"]) if stats["score_std"] is not None else None,
            "score_q25": float(stats["score_q25"]) if stats["score_q25"] is not None else None,
            "score_q75": float(stats["score_q75"]) if stats["score_q75"] is not None else None,
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"Results saved to: {output_path}")


def load_results_from_json(input_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load analysis results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze bucketed JSONL.zst files")
    parser.add_argument("input_dir", type=Path, help="Input directory containing bucket subdirectories")
    parser.add_argument("--workers", type=int, default=128, help="Number of total cores to use (default: 128)")
    parser.add_argument("--output-json", type=Path, default=Path("bucket_analysis_results.json"), 
                       help="Path to save/load results JSON file (default: bucket_analysis_results.json)")
    parser.add_argument("--force-regenerate", action="store_true", 
                       help="Force regeneration of results even if JSON file exists")
    parser.add_argument("--plot-only", action="store_true", 
                       help="Only generate plots from existing JSON results (skip analysis)")

    args = parser.parse_args()

    if not args.plot_only:
        if not args.input_dir.exists():
            print(f"Error: Input directory {args.input_dir} does not exist")
            return 1

        if not args.input_dir.is_dir():
            print(f"Error: {args.input_dir} is not a directory")
            return 1

    # Check if we should load from existing JSON or run analysis
    bucket_stats = None
    
    if args.plot_only:
        # Plot-only mode: load from JSON
        if not args.output_json.exists():
            print(f"Error: JSON file {args.output_json} does not exist. Run analysis first without --plot-only.")
            return 1
        print(f"Loading results from: {args.output_json}")
        bucket_stats = load_results_from_json(args.output_json)
    elif args.output_json.exists() and not args.force_regenerate:
        # JSON exists and not forcing regeneration
        print(f"Found existing results at: {args.output_json}")
        print("Loading existing results. Use --force-regenerate to reanalyze.")
        bucket_stats = load_results_from_json(args.output_json)
    else:
        # Run full analysis
        print(f"Analyzing buckets in: {args.input_dir}")
        bucket_stats = analyze_buckets(args.input_dir, args.workers)
        
        if not bucket_stats:
            print("No buckets found in the input directory")
            return 1
        
        # Save results to JSON
        save_results_to_json(bucket_stats, args.output_json)

    # Print statistics
    print_statistics(bucket_stats)

    # Create visualizations
    create_visualizations(bucket_stats)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
