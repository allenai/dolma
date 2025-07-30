# Repository Quality Filter Tool

A high-performance Rust tool for filtering and analyzing document repositories based on quality scores. This tool processes large datasets of documents organized by programming language and provides sophisticated filtering capabilities based on repository quality metrics.

## Overview

The tool provides four main subcommands that work together to analyze and filter document repositories:

1. **`aggregate`** - Analyzes documents and creates repository quality scores
2. **`bin`** - Groups repositories into quality-based bins using score distributions
3. **`filter`** - Extracts high-quality documents based on bin criteria
4. **`pipeline`** - Runs the complete workflow in a single command

## Installation

```bash
cargo build --release
```

## Quick Start

For the impatient, here's how to get high-quality documents in one command:

```bash
cargo run --release -- pipeline \
  --source-dir /path/to/your/data \
  --target-bins 8,9,10 \
  --output-dir /path/to/filtered_output
```

This will extract documents from the top 30% quality repositories across all programming languages.

---

## Command Reference

### 🔍 `aggregate` - Repository Score Analysis

Analyzes document collections and generates comprehensive quality scores for each repository within each programming language.

#### Usage
```bash
cargo run -- aggregate [OPTIONS] --source-dir <SOURCE_DIR>
```

#### Options
- `--source-dir <SOURCE_DIR>` - Directory containing language subdirectories with `*.jsonl.zst` files
- `--output <OUTPUT>` - Output report path (default: `repo_scores_report` → Arrow format directory)

#### Input Structure
```
data/
├── python/
│   ├── 000.jsonl.zst
│   ├── 001.jsonl.zst
│   └── ...
├── javascript/
│   ├── 000.jsonl.zst
│   └── ...
└── rust/
    └── ...
```

#### Document Format
Each `.jsonl.zst` file should contain JSON lines with this structure:
```json
{
  "id": "doc_123",
  "text": "actual document content here...",
  "metadata": {
    "repo_name": "owner/repository-name",
    "score": 0.85,
    "language": "python"
  }
}
```

#### Example Usage
```bash
# Analyze all languages (creates Arrow/Parquet format - recommended)
cargo run --release -- aggregate --source-dir ./dataset --output ./scores_report

# Process single language directory
cargo run --release -- aggregate --source-dir ./dataset/python --output ./python_scores

# Legacy JSON format (add .json extension to force JSON)
cargo run --release -- aggregate --source-dir ./dataset --output ./scores.json
```

#### Output Formats

##### Arrow/Parquet Format (New - Recommended)
Creates a partitioned directory structure:
```
scores_report/
├── _summary.json                    # Summary statistics
└── language/
    ├── python/
    │   ├── part_0000.parquet       # Repository data partitions
    │   ├── part_0001.parquet       # (enables parallel loading)
    │   └── part_0002.parquet
    ├── javascript/
    │   ├── part_0000.parquet
    │   └── part_0001.parquet
    └── rust/
        └── part_0000.parquet
```

**Benefits**: ~3x smaller, ~10x faster loading, parallel processing

##### Legacy JSON Format
Creates a single JSON file with repository statistics:
```json
{
  "languages": {
    "python": {
      "repo_stats": {
        "numpy/numpy": {
          "document_count": 1500,
          "total_score": 1275.5,
          "average_score": 0.85,
          "min_score": 0.12,
          "max_score": 0.99
        }
      },
      "total_repositories": 5000,
      "total_documents": 150000
    }
  },
  "summary": {
    "total_languages": 10,
    "total_repositories": 50000,
    "total_documents": 2000000
  }
}
```

---

### 📊 `bin` - Quality-Based Repository Binning

Creates quality bins for each programming language using reservoir sampling. Each language gets its own set of bins based on its score distribution.

#### Usage
```bash
cargo run -- bin [OPTIONS]
```

#### Options
- `--report-path <REPORT_PATH>` - Path to score report (directory for Arrow, file for JSON)
- `--source-dir <SOURCE_DIR>` - Source directory (only with `--build-report`)
- `--build-report` - Force rebuild the score report from source data
- `--num-bins <NUM_BINS>` - Number of bins per language (default: 10)
- `--sample-size <SAMPLE_SIZE>` - Sample size per bin (default: 100)
- `--output <OUTPUT>` - Output bin report path (default: `repo_bins_report.json`)

#### Example Usage with Arrow/Parquet Format (Recommended)
```bash
# Use existing Arrow score report (10x faster!)
cargo run --release -- bin --report-path ./scores_report --output ./bins.json

# Create detailed bins with Arrow parallel processing
cargo run --release -- bin \
  --report-path ./scores_report \
  --num-bins 20 \
  --sample-size 200 \
  --output ./detailed_bins.json

# Rebuild Arrow report and create bins
cargo run --release -- bin \
  --source-dir ./dataset \
  --build-report \
  --num-bins 10 \
  --output ./fresh_bins.json
```

#### Example Usage with Legacy JSON Format
```bash
# Use existing JSON score report (automatically detected)
cargo run --release -- bin --report-path ./scores.json --output ./bins.json

# JSON format with streaming processing
cargo run --release -- bin \
  --report-path ./legacy_scores.json \
  --num-bins 20 \
  --sample-size 200 \
  --output ./detailed_bins.json
```

#### Output
Creates language-specific bins:
```json
{
  "language_bins": {
    "python": {
      "language": "python",
      "bins": [
        {
          "min_score": 0.0,
          "max_score": 0.1,
          "total_repos_in_range": 500,
          "sample_repos": [
            {
              "repo_name": "user/low-quality-repo",
              "language": "python",
              "average_score": 0.05,
              "document_count": 10
            }
          ]
        },
        {
          "min_score": 0.9,
          "max_score": 1.0,
          "total_repos_in_range": 50,
          "sample_repos": [
            {
              "repo_name": "numpy/numpy",
              "language": "python",
              "average_score": 0.95,
              "document_count": 1500
            }
          ]
        }
      ]
    }
  }
}
```

---

### 🔍 `filter` - High-Quality Document Extraction

Filters documents based on repository quality bins. Supports both language-specific and cross-language filtering. **Automatically detects Arrow/Parquet vs JSON report formats** for optimal performance.

#### Usage
```bash
cargo run -- filter [OPTIONS] --source-dir <SOURCE_DIR> --report-path <REPORT_PATH> --bin-path <BIN_PATH> --output-dir <OUTPUT_DIR>
```

#### Options
- `--source-dir <SOURCE_DIR>` - Source directory with language subdirectories
- `--report-path <REPORT_PATH>` - Path to score report (directory for Arrow, file for JSON)
- `--bin-path <BIN_PATH>` - Path to bin report file
- `--target-bins <TARGET_BINS>` - Comma-separated bin numbers (1-indexed)
- `--language <LANGUAGE>` - Specific language to filter (optional)
- `--output-dir <OUTPUT_DIR>` - Output directory for filtered files
- `--max-file-size-mb <SIZE>` - Max file size before splitting (default: 50MB)

#### Example Usage with Arrow/Parquet Format (Recommended)
```bash
# Filter top 3 bins from all languages (10x faster with Arrow!)
cargo run --release -- filter \
  --source-dir ./dataset \
  --report-path ./scores_arrow_report \
  --bin-path ./bins.json \
  --target-bins 8,9,10 \
  --output-dir ./high_quality

# Filter Python repositories only with parallel loading
cargo run --release -- filter \
  --source-dir ./dataset \
  --report-path ./scores_arrow_report \
  --bin-path ./bins.json \
  --target-bins 7,8,9,10 \
  --language python \
  --output-dir ./python_high_quality

# Filter with Arrow format detection (automatic)
cargo run --release -- filter \
  --source-dir ./dataset \
  --report-path ./my_report \
  --bin-path ./bins.json \
  --target-bins 9,10 \
  --max-file-size-mb 25 \
  --output-dir ./top_quality
```

#### Example Usage with Legacy JSON Format
```bash
# Filter with JSON format (automatically detected)
cargo run --release -- filter \
  --source-dir ./dataset \
  --report-path ./scores.json \
  --bin-path ./bins.json \
  --target-bins 8,9,10 \
  --output-dir ./high_quality

# JSON format with streaming processing
cargo run --release -- filter \
  --source-dir ./dataset \
  --report-path ./legacy_scores.json \
  --bin-path ./bins.json \
  --target-bins 7,8,9,10 \
  --language python \
  --output-dir ./python_filtered
```

#### Performance Comparison
```bash
# Arrow/Parquet format output:
Using Arrow/Parquet format for repository extraction
  Processing language: python with 2 score ranges
  Loading 4 partition files for python in parallel    # Multi-core loading!
  Completed python - found 1,234 matching repos
  Arrow extraction completed - found 5,678 unique repositories total
Filter completed in 8.5s                              # 10x faster!

# Legacy JSON format output:
Using JSON streaming format for repository extraction
Extracting target repositories from score report (streaming)...
  Processing language 1/12 - found 1,234 repos so far
  Processing language 2/12 - found 2,456 repos so far
  ...
Filter completed in 85.2s                             # Slower but compatible
```

#### Output Structure
Maintains the original language directory structure:
```
high_quality/
├── python/
│   ├── 000000.jsonl.zst
│   ├── 000001.jsonl.zst
│   └── ...
├── javascript/
│   ├── 000000.jsonl.zst
│   └── ...
└── rust/
    └── ...
```

#### Performance Features
- **Parallel processing**: Uses all CPU cores
- **Progress tracking**: Real-time progress bar
- **Memory efficient**: Streams large files
- **Repository ordering**: Documents from same repo stay together
- **Smart chunking**: Splits files without breaking repositories

---

### 🚀 `pipeline` - Complete Workflow

Runs the entire analysis and filtering pipeline in sequence: aggregate → bin → filter.

#### Usage
```bash
cargo run -- pipeline [OPTIONS] --source-dir <SOURCE_DIR> --output-dir <OUTPUT_DIR>
```

#### Options
- `--source-dir <SOURCE_DIR>` - Source directory with language subdirectories
- `--target-bins <TARGET_BINS>` - Comma-separated bin numbers to filter
- `--language <LANGUAGE>` - Specific language to filter (optional)
- `--output-dir <OUTPUT_DIR>` - Output directory for filtered files
- `--num-bins <NUM_BINS>` - Number of bins to create (default: 10)
- `--sample-size <SAMPLE_SIZE>` - Sample size per bin (default: 100)
- `--max-file-size-mb <SIZE>` - Max output file size (default: 50MB)
- `--work-dir <WORK_DIR>` - Directory for intermediate reports (default: `.`)

#### Example Usage
```bash
# Extract top 20% repositories from all languages
cargo run -- pipeline \
  --source-dir ./dataset \
  --target-bins 9,10 \
  --output-dir ./filtered_data

# Extract top 40% from Python only
cargo run -- pipeline \
  --source-dir ./dataset \
  --target-bins 7,8,9,10 \
  --language python \
  --output-dir ./python_filtered \
  --work-dir ./reports

# Create detailed bins and filter top 10%
cargo run -- pipeline \
  --source-dir ./dataset \
  --target-bins 10 \
  --num-bins 20 \
  --sample-size 200 \
  --output-dir ./premium_quality \
  --work-dir ./analysis
```

#### Workflow Output
```
🚀 Starting complete pipeline: aggregate -> bin -> filter
Source: ./dataset
Output: ./filtered_data
Work dir: ./reports

📊 Step 1/3: Aggregating repository scores...
Report will be saved to: ./reports/repo_scores_1703123456.json
[1/12] Processing language directory: python
  ✓ Completed python - 5,000 repos, 150,000 docs
...

🗂️  Step 2/3: Creating score-based bins...
Bin report will be saved to: ./reports/repo_bins_1703123456.json
Creating language-specific bins in parallel...
Processing 12 languages in parallel
...

🔍 Step 3/3: Filtering documents...
Target bins: [9, 10]
Extracting target repositories from score report (streaming)...
Found 5,234 target repositories across 2 bins
...

✅ Pipeline completed in 245.67s
📁 Intermediate reports:
  Score report: ./reports/repo_scores_1703123456.json
  Bin report: ./reports/repo_bins_1703123456.json
📁 Filtered output: ./filtered_data

💡 Tip: You can reuse the reports for additional filtering:
  cargo run -- filter \
    --report-path ./reports/repo_scores_1703123456.json \
    --bin-path ./reports/repo_bins_1703123456.json \
    --target-bins <BINS> \
    --output-dir <OUTPUT>
```

---

## Performance Notes

### Optimization Features
- **Multi-core processing**: Utilizes all available CPU cores
- **Arrow/Parquet format**: Columnar storage with excellent compression and parallel loading
- **Partitioned data**: Files split across CPU cores for maximum parallelism
- **Streaming JSON fallback**: Handles legacy reports without loading everything into memory
- **Progress tracking**: Real-time feedback on processing status
- **Memory efficiency**: Optimized allocations and data structures
- **Binary search**: O(log n) repository binning algorithm

### Storage Formats & Auto-Detection

The tool **automatically detects** whether to use Arrow/Parquet or JSON format based on the report path:

#### Format Detection Rules
- **Directory path** (e.g., `./my_report`) → Arrow/Parquet format
- **File path with .json** (e.g., `./my_report.json`) → JSON format
- **File path without extension** (e.g., `./my_report.txt`) → JSON format

#### Arrow/Parquet Format (Recommended)
New reports use Apache Arrow/Parquet format for superior performance:
- **~10x faster loading**: Columnar format with parallel decompression
- **~3x smaller files**: Better compression than JSON
- **Multi-core reading**: Each partition loaded in parallel
- **Type safety**: Schema-enforced data types
- **Directory structure**: `report_name/language/{lang}/part_XXXX.parquet`

#### Legacy JSON Format
Older reports use streaming JSON processing:
- **Memory efficient**: Streams large files without loading all data
- **Backward compatible**: Works with existing report files
- **Single-threaded**: Sequential processing per language
- **Single file**: `report_name.json`

### Typical Performance

#### With Arrow/Parquet Format
- **Aggregate**: ~1M documents/minute per core → **~4GB partitioned Parquet output**
- **Bin**: ~100K repositories/second (parallel) → **~10x faster with Arrow**
- **Filter**: ~500K documents/minute per core

#### With Legacy JSON Format
- **Aggregate**: ~1M documents/minute per core → **~6.8GB JSON output**
- **Bin**: ~100K repositories/second (streaming) → **Memory limited by JSON size**
- **Filter**: ~500K documents/minute per core

### Memory Requirements
- **Aggregate**: ~500MB per million documents (both formats)
- **Bin (Arrow)**: ~100MB base + parallel partition loading
- **Bin (JSON)**: ~50MB for streaming JSON processing
- **Filter**: ~200MB base + file buffers

---

## Troubleshooting

### Common Issues

**"No repositories found"**
- Check that your documents have `metadata.repo_name` field
- Verify `metadata.score` field exists and is numeric

**"Score report file does not exist"**
- Run `aggregate` command first, or use `--build-report` flag

**Memory issues**
- Use `filter` command instead of `pipeline` for very large datasets
- Reduce `--sample-size` for bin command

**Slow performance**
- Ensure you're using `cargo run --release` for production runs
- Check that source files are compressed (`.jsonl.zst`)

### Getting Help
```bash
# General help
cargo run -- --help

# Command-specific help
cargo run -- aggregate --help
cargo run -- bin --help
cargo run -- filter --help
cargo run -- pipeline --help
```

---

## Advanced Examples

### Multi-Stage Filtering
```bash
# Stage 1: Create initial bins
cargo run -- pipeline \
  --source-dir ./raw_data \
  --target-bins 6,7,8,9,10 \
  --output-dir ./stage1_filtered

# Stage 2: Further refine with different criteria
cargo run -- aggregate --source-dir ./stage1_filtered --output stage1_scores.json
cargo run -- bin --report-path stage1_scores.json --num-bins 5 --output stage1_bins.json
cargo run -- filter \
  --source-dir ./stage1_filtered \
  --report-path stage1_scores.json \
  --bin-path stage1_bins.json \
  --target-bins 4,5 \
  --output-dir ./final_premium
```

### Language-Specific Analysis
```bash
# Analyze each language separately
for lang in python javascript rust go java; do
  cargo run -- pipeline \
    --source-dir ./data \
    --language $lang \
    --target-bins 8,9,10 \
    --output-dir ./filtered_$lang \
    --work-dir ./reports_$lang
done
```

### Custom Quality Thresholds
```bash
# Create fine-grained bins for precise filtering (Arrow format)
cargo run --release -- bin \
  --report-path ./scores_report \
  --num-bins 100 \
  --sample-size 50 \
  --output ./fine_bins.json

# Extract only the top 1% (10x faster with Arrow!)
cargo run --release -- filter \
  --source-dir ./data \
  --report-path ./scores_report \
  --bin-path ./fine_bins.json \
  --target-bins 100 \
  --output-dir ./elite_quality
```

---

## 🚀 Quick Migration Guide: JSON → Arrow

If you have existing JSON reports and want to benefit from Arrow performance:

### Option 1: Convert Existing Data
```bash
# Your existing workflow
cargo run --release -- aggregate --source-dir ./dataset --output ./scores.json
cargo run --release -- bin --report-path ./scores.json --output ./bins.json

# New high-performance workflow
cargo run --release -- aggregate --source-dir ./dataset --output ./scores_arrow
cargo run --release -- bin --report-path ./scores_arrow --output ./bins.json  # 10x faster!
cargo run --release -- filter \
  --source-dir ./dataset \
  --report-path ./scores_arrow \
  --bin-path ./bins.json \
  --target-bins 8,9,10 \
  --output-dir ./filtered
```

### Option 2: Use Pipeline (Recommended)
```bash
# One command, maximum performance
cargo run --release -- pipeline \
  --source-dir ./dataset \
  --target-bins 8,9,10 \
  --output-dir ./filtered \
  --work-dir ./reports
```

### Performance Benefits
- **Aggregate**: Same speed, but creates ~3x smaller output
- **Bin**: ~10x faster with parallel Arrow loading
- **Filter**: ~10x faster repository extraction
- **Storage**: ~70% less disk space used

---

## 📈 Performance Summary

| Operation | JSON Format | Arrow Format | Improvement |
|-----------|-------------|--------------|-------------|
| Report Size | 6.8 GB | 2.3 GB | **3x smaller** |
| Bin Loading | 125s | 12s | **10x faster** |
| Filter Extract | 85s | 8s | **10x faster** |
| Memory Usage | 6.8 GB peak | 200 MB peak | **34x less** |
| CPU Utilization | Single-core | All cores | **Multi-core** |

The Arrow/Parquet format provides dramatic performance improvements while maintaining full backward compatibility with existing JSON workflows.
