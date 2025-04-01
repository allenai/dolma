#!/usr/bin/env python3
import argparse
import gzip
import json
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import msgspec


class SummarySpec(msgspec.Struct):
    """Class to deserialize analyzer output summaries"""
    name: str
    counts: List[int]
    bins: List[float]
    total: int
    sum: float


def load_summaries(file_path: str) -> List[SummarySpec]:
    """Load analyzer summaries from a jsonl.gz file"""
    summaries = []
    
    opener = gzip.open if file_path.endswith('.gz') else open
    with opener(file_path, 'rt') as f:
        for line in f:
            try:
                summary = msgspec.json.decode(line, type=SummarySpec)
                summaries.append(summary)
            except Exception as e:
                print(f"Error decoding line: {e}")
    
    return summaries


def compute_percentiles(
    summary: SummarySpec, 
    buckets: int,
    error_margin: float = 0.05
) -> List[float]:
    """
    Compute percentile thresholds that split the data into approximately equal buckets.
    
    Args:
        summary: Analyzer summary
        buckets: Number of buckets to split the data into
        error_margin: Maximum allowed deviation from equal bucket sizes
        
    Returns:
        List of percentile values that define the bucket boundaries
    """
    # Convert the bins and counts to arrays for easier manipulation
    bins = np.array(summary.bins)
    counts = np.array(summary.counts)
    
    # Calculate cumulative distribution
    cum_counts = np.cumsum(counts)
    total_count = cum_counts[-1]  # Total number of items
    
    # Compute the target size for each bucket (equal distribution)
    target_bucket_size = total_count / buckets
    min_bucket_size = target_bucket_size * (1 - error_margin)
    max_bucket_size = target_bucket_size * (1 + error_margin)
    
    # Calculate the target cumulative counts for each bucket boundary
    target_cum_counts = np.array([i * target_bucket_size for i in range(1, buckets)])
    
    # Find the indices where the cumulative counts exceed the target counts
    boundary_indices = np.searchsorted(cum_counts, target_cum_counts)
    
    # Compute the percentiles by linear interpolation between bin boundaries
    percentiles = []
    for i, idx in enumerate(boundary_indices):
        if idx == 0:  
            # If the target count is before the first bin
            percentile_val = bins[0]
        elif idx >= len(bins):
            # If the target count is after the last bin
            percentile_val = bins[-1]
        else:
            # Get the bin edges and counts for interpolation
            lower_idx = idx - 1
            upper_idx = idx
            
            lower_bin = bins[lower_idx]
            upper_bin = bins[upper_idx]
            
            lower_cum_count = cum_counts[lower_idx]
            upper_cum_count = cum_counts[upper_idx]
            
            # Linear interpolation between bin boundaries
            target_count = target_cum_counts[i]
            if upper_cum_count > lower_cum_count:  # Avoid division by zero
                fraction = (target_count - lower_cum_count) / (upper_cum_count - lower_cum_count)
                percentile_val = lower_bin + fraction * (upper_bin - lower_bin)
            else:
                percentile_val = upper_bin
        
        percentiles.append(percentile_val)
    
    # Verify the bucket sizes
    bucket_sizes = []
    
    # First bucket: from start to first percentile
    first_idx = np.searchsorted(bins, percentiles[0], side='right')
    first_bucket_size = cum_counts[first_idx - 1] if first_idx > 0 else 0
    bucket_sizes.append(first_bucket_size)
    
    # Middle buckets
    for i in range(len(percentiles) - 1):
        lower_idx = np.searchsorted(bins, percentiles[i], side='right')
        upper_idx = np.searchsorted(bins, percentiles[i + 1], side='right')
        
        lower_count = cum_counts[lower_idx - 1] if lower_idx > 0 else 0
        upper_count = cum_counts[upper_idx - 1] if upper_idx > 0 else 0
        
        bucket_size = upper_count - lower_count
        bucket_sizes.append(bucket_size)
    
    # Last bucket: from last percentile to end
    last_idx = np.searchsorted(bins, percentiles[-1], side='right')
    last_count = cum_counts[last_idx - 1] if last_idx > 0 else 0
    last_bucket_size = total_count - last_count
    bucket_sizes.append(last_bucket_size)
    
    # Check if any bucket size is outside the allowed range
    for i, size in enumerate(bucket_sizes):
        if size < min_bucket_size or size > max_bucket_size:
            print(f"Warning: Bucket {i+1} has size {size:,.0f} which is outside the allowed range "
                  f"[{min_bucket_size:,.0f}, {max_bucket_size:,.0f}]")
    
    # Debug info
    print(f"Total count: {total_count:,}")
    print(f"Target bucket size: {target_bucket_size:,.0f}")
    print(f"Actual bucket sizes: {[f'{size:,.0f}' for size in bucket_sizes]}")
    
    # Clean up percentiles - ensure they're strictly increasing and within range
    percentiles = [max(bins[0], min(p, bins[-1])) for p in percentiles]
    
    # Ensure percentiles are strictly increasing
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i-1]:
            percentiles[i] = percentiles[i-1] + (bins[-1] - bins[0]) / (len(bins) * 10)
    
    return percentiles


def generate_mixer_config(
    attribute: str,
    percentiles: List[float],
    documents_path: str,
    output_base_path: str,
    summary: Optional[SummarySpec] = None
) -> Dict[str, Any]:
    """
    Generate a mixer configuration based on computed percentiles.
    
    Args:
        attribute: Name of the attribute to filter on
        percentiles: List of percentile values that define bucket boundaries
        documents_path: Path pattern for input documents
        output_base_path: Base path for output
        summary: Optional summary data to calculate actual document distribution
        
    Returns:
        Dictionary containing the mixer configuration
    """
    # Create buckets with min and max bounds
    all_bounds = [float("-inf")] + percentiles + [float("inf")]
    buckets = list(zip(all_bounds[:-1], all_bounds[1:]))
    
    # Prepare the mixer configuration
    streams = []
    num_buckets_count = len(buckets)
    
    # Calculate actual PMF for each bucket if summary data is available
    actual_pmf = []
    if summary is not None:
        bins = np.array(summary.bins)
        counts = np.array(summary.counts)
        total_count = summary.total
        
        for min_val, max_val in buckets:
            # Find indices where bins fall within this bucket's range
            if min_val == float("-inf"):
                min_idx = 0
            else:
                min_idx = np.searchsorted(bins, min_val, side='left')
                
            if max_val == float("inf"):
                max_idx = len(bins)
            else:
                max_idx = np.searchsorted(bins, max_val, side='left')
            
            # Sum counts in this range
            bucket_count = sum(counts[min_idx:max_idx])
            bucket_pmf = bucket_count / total_count if total_count > 0 else 0
            actual_pmf.append(bucket_pmf)
    
    for i, (min_val, max_val) in enumerate(buckets):
        # Calculate percentile range for this bucket (for informational purposes)
        if min_val == float("-inf"):
            low_percentile = 0
        else:
            low_percentile = (i * 100) // num_buckets_count
            
        if max_val == float("inf"):
            high_percentile = 100
        else:
            high_percentile = ((i + 1) * 100) // num_buckets_count
        
        # 0-indexed bucket name in the format "bucket_i_of_n"
        bucket_name = f"bucket_{i}_of_{num_buckets_count}"
        
        # Create comment about document distribution
        comment = f"Percentile range: {low_percentile}% to {high_percentile}%"
        if actual_pmf and i < len(actual_pmf):
            comment += f", Actual document proportion: {actual_pmf[i]:.2%}"
        
        # Create filters based on min and max values
        filters = {"include": [], "exclude": []}
        
        # Format the values for the filter
        # Special handling for infinity values
        min_str = f"{min_val:.6f}" if min_val != float("-inf") else "-Infinity"
        max_str = f"{max_val:.6f}" if max_val != float("inf") else "Infinity"
        
        # Determine filter attribute name and attribute list name
        filter_attr_name = attribute
        attrs_name = attribute
        
        # Check if this is a nested attribute with double underscore format (like dclm_log__dclm_oh_eli5_log__score)
        # If so, keep the first part for the attributes list and the full path for the filter
        if '__' in attribute:
            attrs_name = attribute.split('__')[0]
            # Keep the full filter_attr_name as is
        
        # Create JSONPath filter for this range using proper syntax
        if min_val == float("-inf") and max_val == float("inf"):
            # All values - since we know the attribute exists, no filter needed
            # Include all documents by setting an empty include filter list
            # filters["include"] is already an empty list, so no need to add anything
            pass
        elif min_val == float("-inf"):
            # Only upper bound - documents with score less than max_val
            filter_expr = f"$.attributes[?(@.{filter_attr_name}[0][2] < {max_str})]"
            filters["include"].append(filter_expr)
        elif max_val == float("inf"):
            # Only lower bound - documents with score greater than or equal to min_val
            filter_expr = f"$.attributes[?(@.{filter_attr_name}[0][2] >= {min_str})]"
            filters["include"].append(filter_expr)
        else:
            # Both bounds - documents with score in range [min_val, max_val)
            filter_expr = f"$.attributes[?(@.{filter_attr_name}[0][2] >= {min_str} && @.{filter_attr_name}[0][2] < {max_str})]"
            filters["include"].append(filter_expr)
        
        # Add a debug print to see the filter expression
        print(f"Bucket {i}: Filter expression: {filter_expr}")
        
        # Prepare attribute list - only include the filter attribute
        attrs = [attrs_name]
        
        # Create the stream configuration
        stream = {
            "name": bucket_name,
            "comment": comment,
            "documents": [documents_path],
            "attributes": attrs,
            "filter": filters,
            "output": {
                "path": f"{output_base_path}/{bucket_name}",
                "max_size_in_bytes": 1000000000  # 1GB default
            }
        }
        
        streams.append(stream)
    
    # Create the full mixer configuration
    config = {
        "streams": streams,
        "processes": 8
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate mixer configuration based on analyzer output")
    parser.add_argument("summary_file", help="Path to the analyzer output summary file (.jsonl.gz)")
    parser.add_argument("config", help="Output path for the mixer configuration")
    parser.add_argument("--buckets", type=int, default=5, help="Number of buckets to split data into")
    parser.add_argument("--error_margin", type=float, default=0.05,
                        help="Maximum allowed deviation from equal bucket sizes (0-1)")
    parser.add_argument("--attribute", help="Name of the attribute to use for filtering")
    parser.add_argument("--documents", default="s3://example-bucket/dolma/documents/*.jsonl.gz",
                        help="Path pattern for input documents")
    parser.add_argument("--output", default="s3://example-bucket/dolma/mixed",
                        help="Base path for output files")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Load summaries from the analyzer output
    summaries = load_summaries(args.summary_file)
    
    if not summaries:
        print(f"No summaries found in {args.summary_file}")
        return
    
    # If attribute name not specified, use the first summary's name
    attribute = args.attribute
    if not attribute:
        # Extract the attribute name from the first summary
        attribute = summaries[0].name
        # Remove the "/score" suffix if present
        if attribute.endswith("/score"):
            attribute = attribute[:-6]
        print(f"Using attribute name: {attribute}")
    
    # Find the summary for the specified attribute
    target_summary = None
    for summary in summaries:
        if summary.name == attribute or summary.name == f"{attribute}/score":
            target_summary = summary
            break
    
    if not target_summary:
        print(f"No summary found for attribute {attribute}")
        return
    
    # Compute percentiles for approximately equal buckets
    percentiles = compute_percentiles(
        target_summary, 
        args.buckets,
        args.error_margin
    )
    
    print(f"Computed percentiles: {[round(p, 4) for p in percentiles]}")
    
    # Generate mixer configuration
    config = generate_mixer_config(
        attribute,
        percentiles,
        args.documents,
        args.output,
        target_summary
    )
    
    # Print debug information about the configuration
    if args.debug:
        print("\nDEBUG: Configuration streams:")
        for i, stream in enumerate(config['streams']):
            print(f"\nStream {i}: {stream['name']}")
            print(f"  Documents: {stream['documents']}")
            print(f"  Attributes: {stream['attributes']}")
            print(f"  Filter include: {stream['filter'].get('include', [])}")
            print(f"  Filter exclude: {stream['filter'].get('exclude', [])}")
            print(f"  Output path: {stream['output']['path']}")
    
    # Store comments to add at the end of the file
    comments = []
    for stream in config['streams']:
        if 'comment' in stream:
            comments.append(f"# {stream['name']}: {stream['comment']}")
            # Remove the comment from the stream config since YAML doesn't support comments in the data
            del stream['comment']
    
    # Customize YAML dumper to handle lists properly
    class ListAsSeqDumper(yaml.SafeDumper):
        pass
    
    # Configure yaml.dump to use this style consistently
    def represent_list(self, data):
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    
    ListAsSeqDumper.add_representer(list, represent_list)
    
    # Write the configuration to a YAML file
    with open(args.config, 'w') as f:
        yaml.dump(config, f, Dumper=ListAsSeqDumper, default_flow_style=False)
        
        # Add bucket information as comments at the end of the file
        if comments:
            f.write("\n# ---- Bucket Information ----\n")
            for comment in comments:
                f.write(f"{comment}\n")
    
    print(f"Mixer configuration written to {args.config}")
    print("Bucket information has been added as comments at the end of the configuration file.")


if __name__ == "__main__":
    main()
