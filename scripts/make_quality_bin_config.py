#!/usr/bin/env python3

# Example command:
# python scripts/make_quality_bin_config.py \
# --attributes 's3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/2shards-dedup/attributes/dclm/global-shard_03_of_10/*/*.jsonl.zstd' \
# --max-files-for-percentiles 1000 -w100 -o stats/t60-b1-r1.0.jsonl \
# --config configs/aw_mix_dclm_baseline_bins_t60-b1-r1.0.yaml -t 60 -b 1 -r 1.0 \
# --documents s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/2shards-dedup/documents/global-shard_03_of_10/*/*.jsonl.zstd \
# --attribute-name dclm__dclm_oh_eli5_log__score \
# --output s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/2shards-dedup/dclm_baseline_bins_t60-b1-r1.0

import argparse
import json
import multiprocessing
import os
import random
import yaml

from typing import Dict, List, Tuple, Optional, Any

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


def compute_attribute_percentiles(
    attr_name: str,
    samples: List[Tuple[float, int]],
    percentiles: List[float],
):
    """
    Compute percentiles for attribute scores.
    
    Args:
        attr_name: Name of the attribute
        samples: List of (score, length) tuples
        percentiles: List of percentiles to compute
    
    Returns:
        Dictionary with percentile statistics and bin boundaries
    """
    if not samples:
        print(f"{attr_name} (0 samples): No data available")
        return {
            "samples": 0,
            "unweighted": {},
            "weighted": {},
        }
    
    # Separate values and weights
    values, weights = zip(*samples)
    values_array = np.array(values)
    weights_array = np.array(weights)
    
    # Compute unweighted percentiles using np.percentile
    unweighted_pct = np.percentile(values_array, percentiles)
    
    # Compute weighted percentiles using np.percentile with weights parameter
    weighted_pct = np.percentile(values_array, percentiles, weights=weights_array, method="inverted_cdf")
    
    print(f"{attr_name} ({len(samples):,} samples):")
    
    # Store percentile results
    stats = {
        "samples": len(samples),
        "unweighted": {},
        "weighted": {}
    }
    
    print("  Unweighted percentiles:")
    for p, val in zip(percentiles, unweighted_pct):
        print(f"    P{p:.3f}: {val:.6f}")
        stats["unweighted"][f"P{p:.3f}"] = val.item()
    
    print("  Weighted percentiles (by document length):")
    for p, val in zip(percentiles, weighted_pct):
        print(f"    P{p:.3f}: {val:.6f}")
        stats["weighted"][f"P{p:.3f}"] = val.item()
    
    return stats


def generate_mixer_config(
    attribute: str,
    bin_percentages: List[float],
    bin_percentiles: List[float],
    documents_path: str,
    sampling_fraction: Optional[float] = None,
    output_path: str,
) -> Dict[str, Any]:
    """
    Generate a mixer configuration based on computed bin boundaries.
    
    Args:
        attribute: Name of the attribute to filter on
        bin_percentages: List of percentages corresponding to the bin boundaries
        bin_boundaries: List of percentile values corresponding to the bin boundaries
        documents_path: Path pattern for input documents
        output_path: Base path for output
        
    Returns:
        Dictionary containing the mixer configuration
    """

    bin_percentages = bin_percentages + [100]
    bin_percentiles = bin_percentiles + [float("inf")]
    
    # Skip everything below the lowest percentile by starting with the first percentile
    # This creates bins like [P50-P60, P60-P70, P70-P80, P80-P90, P90-inf]
    bins = list(zip(bin_percentiles[:-1], bin_percentiles[1:]))
    
    if sampling_fraction is not None:
        bin_widths = [(bins[i][1] - bins[i][0]) / 100 for i in range(len(bins))]
        total_amplify = args.sampling_fraction / sum(bin_widths)
        inverse_bin_widths = [1 / bin_widths[i] for i in range(len(bin_widths))]
        total_inverse_bin_widths = sum(inverse_bin_widths)
        bin_upsamples = [total_amplify * inverse_bin_widths[i] / total_inverse_bin_widths for i in range(len(bins))]
    else:
        bin_upsamples = [1.0] * len(bins)
    
    # Prepare the mixer configuration
    streams = []
    
    for i, (min_val, max_val) in enumerate(bins):
        # 0-indexed bin name in the format "bin_i_of_n" or "pXX_to_pYY"
        # Using percentile-based naming to be more informative
        bin_name = f"p{bin_percentages[i]:.3f}_to_p{bin_percentages[i+1]:.3f}"
        
        # Create filters based on min and max values
        filters = {"include": [], "exclude": []}
        
        # Format the values for the filter
        # Special handling for infinity values
        min_str = f"{min_val:.9f}" 
        max_str = f"{max_val:.9f}" if max_val != float("inf") else "Infinity"
        
        # Determine filter attribute name and attribute list name
        filter_attr_name = attribute
        attrs_name = attribute
        
        # Check if this is a nested attribute with double underscore format
        if '__' in attribute:
            attrs_name = attribute.split('__')[0]
        
        # Create JSONPath filter for this range
        if max_val == float("inf"):
            # Only lower bound
            filter_expr = f"$.attributes[?(@.{filter_attr_name}[0][2] >= {min_str})]"
        else:
            # Both bounds
            filter_expr = f"$.attributes[?(@.{filter_attr_name}[0][2] >= {min_str} && @.{filter_attr_name}[0][2] < {max_str})]"
        
        filters["include"].append(filter_expr)
        
        # Prepare attribute list
        attrs = [attrs_name]
        
        # Create the stream configuration
        stream = {
            "name": bin_name,
            "documents": [documents_path],
            "attributes": attrs,
            "filter": filters,
            "output": {
                "path": f"{output_path}/{bin_name}",
                "max_size_in_bytes": 1073741824  # 1GB default
            },
            "compression": {
                "input": "zst",
                "output": "zst"
            },
            "upsample": sampling_fraction[i]
        }
        
        streams.append(stream)
    
    # Create the full mixer configuration
    config = {
        "streams": streams,
        "processes": 190
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate mixer configuration from attribute files")
    
    # Input options
    parser.add_argument("--attributes", nargs="+", required=True, 
                        help="Paths to attribute files")
    
    # Processing options
    parser.add_argument('-t', '--threshold', type=float, help="Min percentile threshold for bins", default=None)
    parser.add_argument('-b', '--bins', type=int, help="Number of bins", default=None)
    parser.add_argument('-r', '--ratio', type=float, help="bin size ratio", default=None)
    parser.add_argument("-p", "--percentiles", type=float, nargs="+", 
                        default=[50, 60, 70, 80, 90, 95, 99],
                        help="Percentiles to compute and use as bin boundaries (if -t, -b, -r not set)")

    parser.add_argument('-f', '--sampling-fraction', type=float, help="Overall sampling fraction", default=None)

    parser.add_argument("--max-files-for-percentiles", type=int, default=1000,
                        help="Maximum number of files for computing percentiles")
    parser.add_argument("-w", "--num-processes", type=int, default=multiprocessing.cpu_count(), 
                        help="Number of processes to use for computing percentiles")
    parser.add_argument("-o", "--output-stats", type=str,
                        help="Path to output JSON file with statistics")
    
    # Config generation options
    parser.add_argument("--config", type=str,
                        help="Path to output mixer configuration YAML file")
    parser.add_argument("--documents", 
                        default="s3://example-bin/dolma/documents/*.jsonl.gz",
                        help="Path pattern for input documents")
    parser.add_argument("--output", 
                        default="s3://example-bin/dolma/mixed",
                        help="Base path for output files")
    parser.add_argument("--attribute-name", 
                        help="Name of the attribute to use for filtering (defaults to first found)")
    
    args = parser.parse_args()
    
    if args.threshold is not None:
        assert args.bins is not None
        assert args.ratio is not None
        
        if args.ratio != 1.0:
            bin_sum = 1 / (1 - args.ratio)
        else:
            bin_sum = args.bins

        # The first width in the series
        first_width = (100 - args.threshold) / bin_sum
        
        percentiles = [args.threshold]
        
        for i in range(0, args.bins - 1):
            width = first_width * (args.ratio**i)
            percentiles.append(percentiles[-1] + width)
        
        print("Computed percentiles:", percentiles)
    else:
        percentiles = args.percentiles
        print("Percentiles:", percentiles)
    
    
    # Collect attribute samples
    all_samples, total_docs = collect_attribute_samples(
        attributes=args.attributes,
        num_processes=args.num_processes,
        max_files=args.max_files,
    )
    
    if not all_samples:
        print("No attribute samples collected. Exiting.")
        return
    
    # Compute percentiles for all attributes
    stats = {}
    for attr_name, samples in sorted(all_samples.items()):
        # Compute percentiles 
        attr_stats = compute_attribute_percentiles(
            attr_name=attr_name,
            samples=samples,
            percentiles=percentiles,
        )
        stats[attr_name] = attr_stats
    
    # Write statistics to JSON if output is specified
    if args.output_stats:
        output_dir = os.path.dirname(args.output_stats)
        if output_dir:
            mkdir_p(output_dir)
        
        with open(args.output_stats, 'w') as f:
            json.dump({
                "total_documents": total_docs,
                "attributes": stats
            }, f, indent=2)
        
        print(f"Statistics written to {args.output_stats}")
    
    # Generate mixer configuration if requested
    if args.config:
        # Determine which attribute to use for filtering
        attribute_name = args.attribute_name
        
        if not attribute_name:
            # Default to first attribute with samples
            for attr in all_samples:
                if all_samples[attr]:
                    attribute_name = attr
                    break
        
        if not attribute_name or attribute_name not in all_samples:
            print(f"No valid attribute found for configuration generation.")
            return
        

        # Generate the mixer configuration
        config = generate_mixer_config(
            attribute=attribute_name,
            bin_percentages=percentiles,
            bin_percentiles=[stats[attribute_name]["weighted"][f"P{p:.3f}"] for p in percentiles],
            documents_path=args.documents,
            sampling_fraction=args.sampling_fraction,
            output_path=args.output,
        )
        
        # Customize YAML dumper to handle lists properly
        class ListAsSeqDumper(yaml.SafeDumper):
            pass
        
        def represent_list(self, data):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
        ListAsSeqDumper.add_representer(list, represent_list)
        
        # Write the configuration to a YAML file
        with open(args.config, 'w') as f:
            yaml.dump(config, f, Dumper=ListAsSeqDumper, default_flow_style=False)
            
        print(f"Mixer configuration written to {args.config}")


if __name__ == "__main__":
    main() 
