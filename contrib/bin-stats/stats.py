#!/usr/bin/env python3
"""
Analyze bucketed JSONL.zst files and generate distribution plots and statistics.
"""

import argparse
import json
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def count_lines_in_zst_file(file_path: Path) -> int:
    """Count lines in a zstandard compressed JSONL file."""
    try:
        with open(file_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                line_count = 0
                buffer = b''
                while True:
                    chunk = reader.read(8192)
                    if not chunk:
                        break
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if line.strip():
                            line_count += 1
                # Handle last line if no trailing newline
                if buffer.strip():
                    line_count += 1
                return line_count
    except (OSError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        return 0


def extract_scores_from_zst_file(file_path: Path) -> List[float]:
    """Extract metadata.score values from a zstandard compressed JSONL file."""
    scores = []
    try:
        with open(file_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                buffer = b''
                while True:
                    chunk = reader.read(8192)
                    if not chunk:
                        break
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if line.strip():
                            try:
                                record = json.loads(line.decode('utf-8'))
                                if 'metadata' in record and 'score' in record['metadata']:
                                    scores.append(float(record['metadata']['score']))
                            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                                continue
                # Handle last line if no trailing newline
                if buffer.strip():
                    try:
                        record = json.loads(buffer.decode('utf-8'))
                        if 'metadata' in record and 'score' in record['metadata']:
                            scores.append(float(record['metadata']['score']))
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                        pass
    except (OSError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error extracting scores from {file_path}: {e}")
    return scores


def analyze_buckets(input_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Analyze all buckets in the input directory."""
    bucket_stats = {}
    
    print("Scanning buckets...")
    for bucket_dir in input_dir.iterdir():
        if bucket_dir.is_dir():
            bucket_name = bucket_dir.name
            print(f"Processing bucket: {bucket_name}")
            
            total_lines = 0
            all_scores = []
            file_count = 0
            
            # Process all .jsonl.zst files in the bucket
            zst_files = list(bucket_dir.glob("*.jsonl.zst"))
            if not zst_files:
                print(f"  No .jsonl.zst files found in {bucket_name}")
                continue
                
            for file_path in zst_files:
                file_count += 1
                lines = count_lines_in_zst_file(file_path)
                total_lines += lines
                
                scores = extract_scores_from_zst_file(file_path)
                all_scores.extend(scores)
                
                print(f"  {file_path.name}: {lines:,} lines, {len(scores)} scores")
            
            # Create bucket statistics
            base_stats = {
                'total_lines': total_lines,
                'file_count': file_count,
                'scores': all_scores,
            }
            
            if all_scores:
                score_array = np.array(all_scores)
                base_stats.update({
                    'score_min': float(np.min(score_array)),
                    'score_max': float(np.max(score_array)),
                    'score_median': float(np.median(score_array)),
                    'score_mean': float(np.mean(score_array)),
                    'score_std': float(np.std(score_array)),
                    'score_q25': float(np.percentile(score_array, 25)),
                    'score_q75': float(np.percentile(score_array, 75))
                })
            else:
                base_stats.update({
                    'score_min': None,
                    'score_max': None,
                    'score_median': None,
                    'score_mean': None,
                    'score_std': None,
                    'score_q25': None,
                    'score_q75': None
                })
            
            bucket_stats[bucket_name] = base_stats
    
    return bucket_stats


def create_visualizations(bucket_stats: Dict[str, Dict[str, Any]]) -> None:
    """Create distribution plots and visualizations."""
    # Prepare data for plotting
    bucket_names = list(bucket_stats.keys())
    line_counts = [stats['total_lines'] for stats in bucket_stats.values()]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bucket Analysis', fontsize=16)
    
    # 1. Distribution of line counts across buckets
    ax1 = axes[0, 0]
    sns.barplot(x=bucket_names, y=line_counts, ax=ax1)
    ax1.set_title('Lines per Bucket')
    ax1.set_xlabel('Bucket')
    ax1.set_ylabel('Number of Lines')
    ax1.tick_params(axis='x', rotation=45)
    
    # Format y-axis with comma separators
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    
    # 2. Score distribution across all buckets
    ax2 = axes[0, 1]
    all_scores = [
        score for stats in bucket_stats.values() 
        if stats['scores'] for score in stats['scores']
    ]
    
    if all_scores:
        sns.histplot(all_scores, bins=50, ax=ax2)
        ax2.set_title('Score Distribution (All Buckets)')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
    
    # 3. Box plot of scores by bucket
    ax3 = axes[1, 0]
    buckets_with_scores = [
        (bucket_name, stats['scores']) 
        for bucket_name, stats in bucket_stats.items() 
        if stats['scores']
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
        df_melted = df_scores.melt(var_name='Bucket', value_name='Score')
        df_melted = df_melted.dropna(subset=['Score'])
        sns.boxplot(data=df_melted, x='Bucket', y='Score', ax=ax3)
        ax3.set_title('Score Distribution by Bucket')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Scatter plot of median score vs line count
    ax4 = axes[1, 1]
    buckets_with_median = [
        (bucket_name, stats['score_median'], stats['total_lines'])
        for bucket_name, stats in bucket_stats.items()
        if stats['score_median'] is not None
    ]
    
    if buckets_with_median:
        bucket_names_with_scores, median_scores, line_counts_with_scores = zip(*buckets_with_median)
        bucket_names_with_scores = list(bucket_names_with_scores)
        median_scores = list(median_scores)
        line_counts_with_scores = list(line_counts_with_scores)
    
    if buckets_with_median:
        ax4.scatter(line_counts_with_scores, median_scores, alpha=0.7)
        ax4.set_title('Median Score vs Line Count')
        ax4.set_xlabel('Lines in Bucket')
        ax4.set_ylabel('Median Score')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        
        # Add bucket labels to points
        for bucket_name, median_score, line_count in zip(bucket_names_with_scores, median_scores, line_counts_with_scores):
            ax4.annotate(bucket_name, (line_count, median_score), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display plots: {e}")
        print("Plots may not be available in headless environments")


def print_statistics(bucket_stats: Dict[str, Dict[str, Any]]) -> None:
    """Print detailed statistics about the buckets."""
    print("\n" + "="*80)
    print("BUCKET ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    total_lines = sum(stats['total_lines'] for stats in bucket_stats.values())
    total_files = sum(stats['file_count'] for stats in bucket_stats.values())
    total_buckets = len(bucket_stats)
    
    print(f"Total buckets: {total_buckets}")
    print(f"Total files: {total_files:,}")
    print(f"Total lines: {total_lines:,}")
    if total_buckets > 0:
        print(f"Average lines per bucket: {total_lines / total_buckets:,.0f}")
    else:
        print("Average lines per bucket: 0")
    
    # Sort buckets by line count
    sorted_buckets = sorted(bucket_stats.items(), key=lambda x: x[1]['total_lines'], reverse=True)
    
    print(f"\nBUCKET DETAILS (sorted by line count):")
    print("-" * 80)
    print(f"{'Bucket':<15} {'Lines':<12} {'Files':<8} {'Score Stats':<30}")
    print("-" * 80)
    
    for bucket_name, stats in sorted_buckets:
        score_info = ""
        if stats['score_median'] is not None:
            score_info = f"min:{stats['score_min']:.3f} med:{stats['score_median']:.3f} max:{stats['score_max']:.3f}"
        else:
            score_info = "No scores found"
        
        print(f"{bucket_name:<15} {stats['total_lines']:<12,} {stats['file_count']:<8} {score_info:<30}")
    
    # Score analysis
    print(f"\nSCORE ANALYSIS:")
    print("-" * 80)
    
    buckets_with_scores = [
        (name, stats) for name, stats in bucket_stats.items() 
        if stats['score_median'] is not None
    ]
    
    if buckets_with_scores:
        print(f"{'Bucket':<15} {'Count':<10} {'Mean':<8} {'Std':<8} {'Q25':<8} {'Q75':<8}")
        print("-" * 80)
        
        for bucket_name, stats in sorted(buckets_with_scores, key=lambda x: x[1]['score_median'], reverse=True):
            print(f"{bucket_name:<15} {len(stats['scores']):<10,} "
                  f"{stats['score_mean']:<8.3f} {stats['score_std']:<8.3f} "
                  f"{stats['score_q25']:<8.3f} {stats['score_q75']:<8.3f}")
        
        # Correlation analysis
        if len(buckets_with_scores) > 1:
            line_counts = [stats['total_lines'] for _, stats in buckets_with_scores]
            median_scores = [stats['score_median'] for _, stats in buckets_with_scores]
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
    parser = argparse.ArgumentParser(description='Analyze bucketed JSONL.zst files')
    parser.add_argument('input_dir', type=Path, help='Input directory containing bucket subdirectories')
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        return 1
    
    print(f"Analyzing buckets in: {args.input_dir}")
    
    # Analyze buckets
    bucket_stats = analyze_buckets(args.input_dir)
    
    if not bucket_stats:
        print("No buckets found in the input directory")
        return 1
    
    # Print statistics
    print_statistics(bucket_stats)
    
    # Create visualizations
    create_visualizations(bucket_stats)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())