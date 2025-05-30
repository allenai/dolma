#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

def calculate_bin_boundaries(threshold, num_bins, ratio):
    """Calculate bin boundaries based on threshold, number of bins, and ratio."""
    if ratio != 1.0:
        bin_sum = 1 / (1 - ratio)
    else:
        bin_sum = num_bins

    # The first width in the series
    first_width = (100 - threshold) / bin_sum
    
    percentiles = [threshold]
    
    for i in range(0, num_bins - 1):
        width = first_width * (ratio**i)
        percentiles.append(percentiles[-1] + width)
    
    return percentiles

def calculate_upsamples(percentiles, sampling_fraction):
    """Calculate upsample factors for each bin."""
    percentage_bins = list(zip(percentiles, percentiles[1:] + [100]))
    bin_widths = [(percentage_bins[i][1] - percentage_bins[i][0]) / 100 for i in range(len(percentage_bins))]
    
    # Calculate bin upsamples to ensure equal area per bin
    num_bins = len(bin_widths)
    bin_upsamples = [sampling_fraction / num_bins / bin_widths[i] for i in range(num_bins)]
    
    return bin_widths, bin_upsamples

def plot_upsampling_distribution(threshold, num_bins, ratio, sampling_fraction):
    """Plot the upsampling distribution."""
    # Calculate bin boundaries and upsamples
    percentiles = calculate_bin_boundaries(threshold, num_bins, ratio)
    bin_widths, bin_upsamples = calculate_upsamples(percentiles, sampling_fraction)
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Bin boundaries and widths
    ax1.bar(percentiles[:-1], bin_widths, width=np.diff(percentiles), align='edge')
    ax1.set_xlabel('Percentile')
    ax1.set_ylabel('Bin Width')
    ax1.set_title('Bin Width Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Upsample factors
    ax2.bar(percentiles[:-1], bin_upsamples, width=np.diff(percentiles), align='edge')
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('Upsample Factor')
    ax2.set_title('Upsample Factor Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, (p, w, u) in enumerate(zip(percentiles[:-1], bin_widths, bin_upsamples)):
        ax1.text(p + w/2, w/2, f'Bin {i+1}\n{w:.2f}', ha='center', va='center')
        ax2.text(p + w/2, u/2, f'Bin {i+1}\n{u:.2f}x', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize upsampling distribution')
    parser.add_argument('-t', '--threshold', type=float, required=True,
                        help='Min percentile threshold for bins')
    parser.add_argument('-b', '--num-bins', type=int, required=True,
                        help='Number of bins')
    parser.add_argument('-r', '--ratio', type=float, required=True,
                        help='Bin size ratio')
    parser.add_argument('-f', '--sampling-fraction', type=float, required=True,
                        help='Overall sampling fraction')
    
    args = parser.parse_args()
    
    plot_upsampling_distribution(
        threshold=args.threshold,
        num_bins=args.num_bins,
        ratio=args.ratio,
        sampling_fraction=args.sampling_fraction
    )

if __name__ == '__main__':
    main() 