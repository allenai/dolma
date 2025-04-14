import argparse
import json
import multiprocessing
import os
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np
import smart_open
import tqdm
from msgspec.json import Decoder

from dolma.core.data_types import OutputSpec
from dolma.core.paths import glob_path, mkdir_p


def process_file(source_path: str):
    """Process a single file and extract attribute samples."""
    
    # Instantiate a decoder for faster decoding
    decoder = Decoder(OutputSpec)
    
    # Dictionary to store attribute samples
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


def main(
    attributes: List[str],
    percentiles: List[float],
    num_processes: int = 1,
    output: Optional[str] = None,
):
    """Main function to sample attributes and compute percentiles."""
    
    # Find all attribute files
    print("Collecting attribute files...")
    input_files = []
    for attr_pattern in attributes:
        input_files.extend(
            glob_path(attr_pattern, autoglob_dirs=True, recursive_dirs=True, yield_dirs=False)
        )
    
    if not input_files:
        print("No attribute files found!")
        return
    
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
    
    # Compute and display percentiles
    print("\nPercentile Analysis:")
    print("-" * 80)
    
    stats = {}
    
    for attr_name, samples in sorted(all_samples.items()):
        if not samples:
            print(f"{attr_name} (0 samples): No data available")
            stats[attr_name] = {"samples": 0, "unweighted": {}, "weighted": {}}
            continue
        
        # Separate values and weights
        values, weights = zip(*samples)
        values_array = np.array(values)
        weights_array = np.array(weights)
        
        # Compute unweighted percentiles using np.percentile
        unweighted_pct = np.percentile(values_array, percentiles)
        
        # Compute weighted percentiles using np.percentile with weights parameter
        weighted_pct = np.percentile(values_array, percentiles, weights=weights_array, method="inverted_cdf")
        
        print(f"{attr_name} ({len(samples):,} samples):")
        print("  Unweighted percentiles:")
        
        unweighted_stats = {}
        for p, val in zip(percentiles, unweighted_pct):
            print(f"    P{p}: {val:.6f}")
            unweighted_stats[f"P{p}"] = val.item()  # Convert numpy value to Python native type
        
        print("  Weighted percentiles (by document length):")
        
        weighted_stats = {}
        for p, val in zip(percentiles, weighted_pct):
            print(f"    P{p}: {val:.6f}")
            weighted_stats[f"P{p}"] = val.item()  # Convert numpy value to Python native type
        
        print()
        
        stats[attr_name] = {
            "samples": len(samples),
            "unweighted": unweighted_stats,
            "weighted": weighted_stats
        }
    
    # Write statistics to JSON if output is specified
    if output:
        output_dir = os.path.dirname(output)
        if output_dir:
            mkdir_p(output_dir)
        
        with open(output, 'w') as f:
            json.dump({
                "total_documents": total_docs,
                "attributes": stats
            }, f, indent=2)
        
        print(f"Statistics written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample attributes and compute percentiles")
    parser.add_argument("--attributes", nargs="+", help="Paths to attribute files")
    parser.add_argument("-w", "--num-processes", type=int, default=multiprocessing.cpu_count(), 
                        help="Number of processes to use")
    parser.add_argument("-p", "--percentiles", type=float, nargs="+", default=[50, 60, 70, 80, 90],
                        help="percentiles to compute")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to output JSON file")
    
    args = parser.parse_args()
    main(
        attributes=args.attributes,
        percentiles=args.percentiles,
        num_processes=args.num_processes,
        output=args.output,
    )
