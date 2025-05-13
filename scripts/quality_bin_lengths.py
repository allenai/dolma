#!/usr/bin/env python3

import argparse
import json
import multiprocessing
import os
import random
from typing import Dict, List, Tuple, Optional, Any

import msgspec
import numpy as np
import smart_open
import tqdm
from msgspec.json import Decoder

from dolma.core.data_types import OutputSpec
from dolma.core.paths import glob_path, mkdir_p


def process_file(source_path: str):
    """Process a single file and extract attribute samples and document lengths."""
    
    # Instantiate a decoder for faster decoding
    decoder = Decoder(OutputSpec)
    
    # Dictionary to store attribute samples and lengths
    attr_samples: Dict[str, List[Tuple[float, int]]] = {}
    
    # Count of processed documents
    docs_cnt = 0
    
    try:
        with smart_open.open(source_path) as f:
            for ln in f:
                try:
                    row = decoder.decode(ln)
                except Exception:
                    continue
                
                # Sample attributes
                for attr_name, attr_values in row.attributes.items():
                    # Skip empty attributes
                    if not attr_values:
                        continue
                    
                    # Process scores and lengths
                    for start, end, score in attr_values:
                        # Track scores with document length for weighting
                        score_key = f"{attr_name}"
                        if score_key not in attr_samples:
                            attr_samples[score_key] = []
                        attr_samples[score_key].append((score, end - start))
                
                # Increment document count
                docs_cnt += 1
    except Exception as e:
        print(f"Error processing {source_path}: {str(e)}")
    
    return attr_samples, docs_cnt


def collect_attribute_samples(
    attributes: List[str],
    num_processes: int = 1,
    max_files: Optional[int] = None,
):
    """Collect attribute samples from files."""
    
    # Find all attribute files
    print("Collecting attribute files...")
    input_files = []
    for attr_pattern in attributes:
        input_files.extend(
            glob_path(attr_pattern, autoglob_dirs=True, recursive_dirs=True, yield_dirs=False)
        )
    
    random.seed(42)
    random.shuffle(input_files)
    if max_files:
        input_files = input_files[:max_files]
    
    if not input_files:
        print("No attribute files found!")
        return {}, 0
    
    print(f"Found {len(input_files)} attribute files")
    
    # Process files with multiprocessing
    print(f"Processing attribute files with {num_processes} processes...")
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Use tqdm to show progress
    results = []
    
    for result in tqdm.tqdm(
        pool.imap_unordered(process_file, input_files), 
        total=len(input_files),
        desc="Processing files",
        unit=" files"
    ):
        results.append(result)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Merge results
    print("Merging results...")
    all_samples: Dict[str, List[Tuple[float, int]]] = {}
    total_docs = 0
    
    for attr_samples, docs_cnt in results:
        for attr_name, samples in attr_samples.items():
            if attr_name not in all_samples:
                all_samples[attr_name] = []
            all_samples[attr_name].extend(samples)
        total_docs += docs_cnt
    
    print(f"Processed {total_docs:,} documents and collected samples for {len(all_samples)} attributes")
    
    return all_samples, total_docs


def compute_bin_length_stats(
    attr_name: str,
    samples: List[Tuple[float, int]],
    percentiles: List[float],
):
    """
    Compute length statistics for each quality bin.
    
    Args:
        attr_name: Name of the attribute
        samples: List of (score, length) tuples
        percentiles: List of percentiles to use as bin boundaries
    
    Returns:
        Dictionary with length statistics for each bin
    """
    if not samples:
        print(f"{attr_name} (0 samples): No data available")
        return {
            "samples": 0,
            "bins": []
        }
    
    # Separate values and lengths
    values, lengths = zip(*samples)
    values_array = np.array(values)
    lengths_array = np.array(lengths)
    
    # Compute unweighted percentiles for bin boundaries
    bin_boundaries = np.percentile(values_array, percentiles)
    
    # Initialize results
    stats = {
        "samples": len(samples),
        "bins": []
    }
    
    # Process each bin
    for i in range(len(bin_boundaries)):
        # Get the range for this bin
        min_val = bin_boundaries[i]
        max_val = bin_boundaries[i + 1] if i < len(bin_boundaries) - 1 else float('inf')
        
        # Find documents in this bin
        mask = (values_array >= min_val) & (values_array < max_val)
        bin_lengths = lengths_array[mask]
        
        if len(bin_lengths) > 0:
            # Compute length statistics
            length_stats = {
                "min_score": min_val,
                "max_score": max_val,
                "documents": len(bin_lengths),
                "median_length": float(np.median(bin_lengths)),
                "q1_length": float(np.percentile(bin_lengths, 25)),
                "q3_length": float(np.percentile(bin_lengths, 75)),
                "min_length": int(np.min(bin_lengths)),
                "max_length": int(np.max(bin_lengths)),
                "mean_length": float(np.mean(bin_lengths)),
                "std_length": float(np.std(bin_lengths))
            }
        else:
            length_stats = {
                "min_score": min_val,
                "max_score": max_val,
                "documents": 0,
                "median_length": None,
                "q1_length": None,
                "q3_length": None,
                "min_length": None,
                "max_length": None,
                "mean_length": None,
                "std_length": None
            }
        
        stats["bins"].append(length_stats)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze document lengths within quality bins")
    
    # Input options
    parser.add_argument("--attributes", nargs="+", required=True, 
                        help="Paths to attribute files")
    
    # Processing options
    parser.add_argument('-t', '--threshold', type=float, help="Min percentile threshold for bins", default=None)
    parser.add_argument('-b', '--num-bins', type=int, help="Number of bins", default=None)
    parser.add_argument('-r', '--ratio', type=float, help="bin size ratio", default=None)
    parser.add_argument("-p", "--percentiles", type=float, nargs="+", 
                        default=[50, 60, 70, 80, 90, 95, 99],
                        help="Percentiles to compute and use as bin boundaries (if -t, -b, -r not set)")

    parser.add_argument("--max-files-for-percentiles", type=int, default=1000,
                        help="Maximum number of files for computing percentiles")
    parser.add_argument("--num-processes-for-percentiles", type=int, default=multiprocessing.cpu_count(), 
                        help="Number of processes to use for computing percentiles")
    parser.add_argument("-o", "--output-stats", type=str, required=True,
                        help="Path to output JSON file with statistics")
    parser.add_argument("--attribute-name", 
                        help="Name of the attribute to analyze (defaults to first found)")
    
    args = parser.parse_args()
    
    if args.threshold is not None:
        assert args.num_bins is not None
        assert args.ratio is not None
        
        if args.ratio != 1.0:
            bin_sum = 1 / (1 - args.ratio)
        else:
            bin_sum = args.num_bins

        # The first width in the series
        first_width = (100 - args.threshold) / bin_sum
        
        percentiles = [args.threshold]
        
        for i in range(0, args.num_bins - 1):
            width = first_width * (args.ratio**i)
            percentiles.append(percentiles[-1] + width)
        
        print("Computed percentiles:", percentiles)
    else:
        percentiles = args.percentiles
        print("Percentiles:", percentiles)
    
    # Collect attribute samples
    all_samples, total_docs = collect_attribute_samples(
        attributes=args.attributes,
        num_processes=args.num_processes_for_percentiles,
        max_files=args.max_files_for_percentiles,
    )
    
    if not all_samples:
        print("No attribute samples collected. Exiting.")
        return
    
    stats = {}
    if args.attribute_name:
        # Compute stats for the target attribute
        attr_name = args.attribute_name
        samples = all_samples[attr_name]
        attr_stats = compute_bin_length_stats(
            attr_name=attr_name,
            samples=samples,
            percentiles=percentiles,
        )
        stats[attr_name] = attr_stats
    else:
        # If no attribute name was specified, compute for all collected attributes
        for attr_name, samples in sorted(all_samples.items()):
            attr_stats = compute_bin_length_stats(
                attr_name=attr_name,
                samples=samples,
                percentiles=percentiles,
            )
            stats[attr_name] = attr_stats
    
    # Write statistics to JSON
    output_dir = os.path.dirname(args.output_stats)
    if output_dir:
        mkdir_p(output_dir)
    
    with open(args.output_stats, 'w') as f:
        json.dump({
            "total_documents": total_docs,
            "attributes": stats
        }, f, indent=2)
    
    print(f"Statistics written to {args.output_stats}")


if __name__ == "__main__":
    main()
