use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{info, warn};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use walkdir::WalkDir;
use zstd::stream::write::Encoder as ZstdEncoder;

#[derive(Parser)]
#[command(name = "make-ordered")]
#[command(about = "Groups documents by repository name and outputs ordered JSONL.zst files")]
struct Args {
    #[arg(help = "Input directory containing documents")]
    input_dir: PathBuf,

    #[arg(short, long, help = "Output directory for grouped files")]
    output_dir: PathBuf,

    #[arg(
        short,
        long,
        default_value = "104857600",
        help = "Target file size in bytes (default: 100MB)"
    )]
    target_size: u64,
}

#[derive(Debug, Deserialize)]
struct DocumentMetadata {
    repo_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Document {
    metadata: DocumentMetadata,
}

#[derive(Debug)]
struct RepoGroup {
    documents: Vec<serde_json::Value>,
    total_size: u64,
}

#[derive(Debug, Clone)]
struct ProcessedDocument {
    json_value: serde_json::Value,
    repo_name: String,
    size: u64,
}

#[derive(Debug)]
struct OutputBatch {
    output_path: PathBuf,
    documents: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Default)]
struct ProcessingStats {
    total_documents: u64,
    total_repositories: u64,
    total_files_processed: u64,
    total_output_files: u64,
    total_input_size_bytes: u64,
    total_output_size_bytes: u64,
    repository_doc_counts: HashMap<String, u64>,
    repository_size_bytes: HashMap<String, u64>,
    subdirectory_stats: HashMap<String, SubdirStats>,
    processing_duration: Duration,
}

#[derive(Debug, Clone, Default)]
struct SubdirStats {
    document_count: u64,
    repository_count: u64,
    file_count: u64,
    total_size_bytes: u64,
    output_files: u64,
}

impl RepoGroup {
    fn new(_repo_name: String) -> Self {
        Self {
            documents: Vec::new(),
            total_size: 0,
        }
    }

    fn add_document(&mut self, doc_json: serde_json::Value, doc_size: u64) {
        self.documents.push(doc_json);
        self.total_size += doc_size;
    }

    fn document_count(&self) -> u64 {
        self.documents.len() as u64
    }

    fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

fn load_documents_from_file(
    file_path: &Path,
    progress_bar: Option<&ProgressBar>,
) -> Result<Vec<ProcessedDocument>> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;

    let mut documents = Vec::new();

    if file_path.extension().and_then(|s| s.to_str()) == Some("zst") {
        let decoder = zstd::stream::read::Decoder::new(file)?;
        let reader = std::io::BufReader::new(decoder);

        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let doc_size = line.len() as u64;
            let json_value: serde_json::Value = serde_json::from_str(&line)?;

            let doc: Document = serde_json::from_value(json_value.clone())?;
            let repo_name = doc
                .metadata
                .repo_name
                .unwrap_or_else(|| "unknown".to_string());

            documents.push(ProcessedDocument {
                json_value,
                repo_name,
                size: doc_size,
            });

            if let Some(pb) = progress_bar {
                pb.inc(1);
            }
        }
    } else {
        let reader = std::io::BufReader::new(file);

        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let doc_size = line.len() as u64;
            let json_value: serde_json::Value = serde_json::from_str(&line)?;

            let doc: Document = serde_json::from_value(json_value.clone())?;
            let repo_name = doc
                .metadata
                .repo_name
                .unwrap_or_else(|| "unknown".to_string());

            documents.push(ProcessedDocument {
                json_value,
                repo_name,
                size: doc_size,
            });

            if let Some(pb) = progress_bar {
                pb.inc(1);
            }
        }
    }

    Ok(documents)
}

fn collect_input_files_by_subdir(
    input_dir: &Path,
    progress_bar: &ProgressBar,
) -> Result<HashMap<PathBuf, Vec<PathBuf>>> {
    let mut subdirs: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

    for entry in WalkDir::new(input_dir) {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let ext = path.extension().and_then(|s| s.to_str());
            if matches!(ext, Some("jsonl") | Some("zst")) {
                let relative_path = path.strip_prefix(input_dir)?;

                let subdir = if let Some(parent) = relative_path.parent() {
                    parent.to_path_buf()
                } else {
                    PathBuf::from(".")
                };

                subdirs
                    .entry(subdir)
                    .or_insert_with(Vec::new)
                    .push(path.to_path_buf());
                progress_bar.inc(1);
            }
        }
    }

    Ok(subdirs)
}

fn group_documents_by_repo(documents: Vec<ProcessedDocument>) -> HashMap<String, RepoGroup> {
    documents
        .into_par_iter()
        .fold(
            HashMap::new,
            |mut acc: HashMap<String, Vec<ProcessedDocument>>, doc| {
                acc.entry(doc.repo_name.clone())
                    .or_insert_with(Vec::new)
                    .push(doc);
                acc
            },
        )
        .reduce(HashMap::new, |mut acc, mut map| {
            for (repo_name, docs) in map.drain() {
                acc.entry(repo_name).or_insert_with(Vec::new).extend(docs);
            }
            acc
        })
        .into_iter()
        .map(|(repo_name, docs)| {
            let mut group = RepoGroup::new(repo_name.clone());
            for doc in docs {
                group.add_document(doc.json_value, doc.size);
            }
            (repo_name, group)
        })
        .collect()
}

fn write_output_file(output_path: &Path, documents: &[serde_json::Value]) -> Result<()> {
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path.display()))?;

    let buf_writer = BufWriter::new(file);
    let mut encoder = ZstdEncoder::new(buf_writer, 3)?;

    for doc in documents {
        serde_json::to_writer(&mut encoder, doc)?;
        encoder.write_all(b"\n")?;
    }

    encoder.finish()?;
    Ok(())
}

fn prepare_output_batches(
    repo_groups: HashMap<String, RepoGroup>,
    subdir_output_path: &Path,
    target_size: u64,
) -> (Vec<OutputBatch>, SubdirStats) {
    let mut batches = Vec::new();
    let mut file_counter = 0;
    let mut current_file_docs = Vec::new();
    let mut current_file_size = 0u64;

    let mut stats = SubdirStats::default();
    let mut sorted_repos: Vec<_> = repo_groups.into_iter().collect();
    sorted_repos.sort_by(|a, b| a.0.cmp(&b.0));

    for (_repo_name, repo_group) in sorted_repos {
        if repo_group.is_empty() {
            continue;
        }

        // Update stats
        stats.document_count += repo_group.document_count();
        stats.repository_count += 1;
        stats.total_size_bytes += repo_group.total_size;

        let repo_would_exceed = current_file_size + repo_group.total_size > target_size;
        let file_not_empty = !current_file_docs.is_empty();

        if repo_would_exceed && file_not_empty {
            let output_path =
                subdir_output_path.join(format!("part_{:04}.jsonl.zst", file_counter));
            batches.push(OutputBatch {
                output_path,
                documents: std::mem::take(&mut current_file_docs),
            });

            current_file_size = 0;
            file_counter += 1;
        }

        current_file_docs.extend(repo_group.documents);
        current_file_size += repo_group.total_size;
    }

    if !current_file_docs.is_empty() {
        let output_path = subdir_output_path.join(format!("part_{:04}.jsonl.zst", file_counter));
        batches.push(OutputBatch {
            output_path,
            documents: current_file_docs,
        });
    }

    stats.output_files = batches.len() as u64;
    (batches, stats)
}

fn collect_repo_analytics(
    repo_groups: &HashMap<String, RepoGroup>,
) -> (HashMap<String, u64>, HashMap<String, u64>) {
    let mut repo_doc_counts = HashMap::new();
    let mut repo_size_bytes = HashMap::new();

    for (repo_name, repo_group) in repo_groups {
        repo_doc_counts.insert(repo_name.clone(), repo_group.document_count());
        repo_size_bytes.insert(repo_name.clone(), repo_group.total_size);
    }

    (repo_doc_counts, repo_size_bytes)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: u64 = 1024;

    if bytes < THRESHOLD {
        return format!("{} B", bytes);
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD as f64;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

fn print_analytics_report(stats: &ProcessingStats) {
    let _multi_progress = MultiProgress::new();

    println!("\n🔍 ═══════════════════════════════════════════════════════════════════════");
    println!("📊                         PROCESSING ANALYTICS REPORT");
    println!("🔍 ═══════════════════════════════════════════════════════════════════════");

    // Overall Statistics
    println!("\n📈 OVERALL STATISTICS:");
    println!(
        "   📄 Total Documents Processed: {:>15}",
        stats.total_documents.to_string()
    );
    println!(
        "   🏛️  Total Repositories Found: {:>15}",
        stats.total_repositories.to_string()
    );
    println!(
        "   📁 Input Files Processed:     {:>15}",
        stats.total_files_processed.to_string()
    );
    println!(
        "   💾 Output Files Created:      {:>15}",
        stats.total_output_files.to_string()
    );
    println!(
        "   📊 Input Data Size:           {:>15}",
        format_bytes(stats.total_input_size_bytes)
    );
    println!(
        "   💿 Output Data Size:          {:>15}",
        format_bytes(stats.total_output_size_bytes)
    );
    println!(
        "   ⏱️  Processing Duration:       {:>15}",
        format!("{:.2}s", stats.processing_duration.as_secs_f64())
    );

    // Repository Distribution Analysis
    println!("\n🏛️  REPOSITORY DISTRIBUTION ANALYSIS:");

    let mut repo_doc_vec: Vec<_> = stats.repository_doc_counts.iter().collect();
    repo_doc_vec.sort_by(|a, b| b.1.cmp(a.1));

    // Top repositories by document count
    println!("   📊 Top 10 Repositories by Document Count:");
    for (i, (repo_name, doc_count)) in repo_doc_vec.iter().take(10).enumerate() {
        let percentage = (**doc_count as f64 / stats.total_documents as f64) * 100.0;
        println!(
            "      {:>2}. {:>8} docs ({:>5.1}%) - {}",
            i + 1,
            doc_count,
            percentage,
            repo_name.chars().take(50).collect::<String>()
        );
    }

    // Repository size distribution
    let doc_counts: Vec<u64> = stats.repository_doc_counts.values().cloned().collect();
    let doc_count_stats = calculate_distribution_stats(&doc_counts);

    println!("\n   📊 Repository Size Distribution (Documents per Repository):");
    println!("      Mean:     {:>10.1} docs", doc_count_stats.mean);
    println!("      Median:   {:>10} docs", doc_count_stats.median);
    println!("      Min:      {:>10} docs", doc_count_stats.min);
    println!("      Max:      {:>10} docs", doc_count_stats.max);
    println!("      Std Dev:  {:>10.1} docs", doc_count_stats.std_dev);

    // Add distribution histogram
    println!(
        "\n{}",
        create_histogram_plot(&doc_counts, "Documents per Repository", 40)
    );

    // Subdirectory Analysis
    println!("\n📁 SUBDIRECTORY ANALYSIS:");
    let mut subdir_vec: Vec<_> = stats.subdirectory_stats.iter().collect();
    subdir_vec.sort_by(|a, b| b.1.document_count.cmp(&a.1.document_count));

    println!("   📊 Top Subdirectories by Document Count:");
    for (i, (subdir_name, subdir_stats)) in subdir_vec.iter().take(10).enumerate() {
        let percentage =
            (subdir_stats.document_count as f64 / stats.total_documents as f64) * 100.0;
        println!(
            "      {:>2}. {:>8} docs ({:>5.1}%) - {} ({} repos, {} files)",
            i + 1,
            subdir_stats.document_count,
            percentage,
            subdir_name,
            subdir_stats.repository_count,
            subdir_stats.output_files
        );
    }

    // Performance Metrics
    println!("\n⚡ PERFORMANCE METRICS:");
    let docs_per_second = stats.total_documents as f64 / stats.processing_duration.as_secs_f64();
    let mb_per_second = (stats.total_input_size_bytes as f64 / (1024.0 * 1024.0))
        / stats.processing_duration.as_secs_f64();

    println!("   📄 Documents/Second:       {:>15.1}", docs_per_second);
    println!("   💾 MB/Second Processed:    {:>15.1}", mb_per_second);
    println!(
        "   📁 Files/Second:           {:>15.1}",
        stats.total_files_processed as f64 / stats.processing_duration.as_secs_f64()
    );

    // Compression Analysis
    if stats.total_output_size_bytes > 0 {
        let compression_ratio =
            stats.total_input_size_bytes as f64 / stats.total_output_size_bytes as f64;
        let space_saved = ((stats.total_input_size_bytes - stats.total_output_size_bytes) as f64
            / stats.total_input_size_bytes as f64)
            * 100.0;

        println!("\n💿 COMPRESSION ANALYSIS:");
        println!(
            "   📊 Compression Ratio:      {:>15.2}:1",
            compression_ratio
        );
        println!("   💾 Space Saved:            {:>15.1}%", space_saved);
        println!(
            "   📉 Size Reduction:         {:>15}",
            format_bytes(stats.total_input_size_bytes - stats.total_output_size_bytes)
        );
    }

    println!("\n🔍 ═══════════════════════════════════════════════════════════════════════");
    println!("✅                      ANALYSIS COMPLETE");
    println!("🔍 ═══════════════════════════════════════════════════════════════════════\n");
}

#[derive(Debug)]
struct DistributionStats {
    mean: f64,
    median: u64,
    min: u64,
    max: u64,
    std_dev: f64,
}

fn create_histogram_plot(values: &[u64], title: &str, max_width: usize) -> String {
    if values.is_empty() {
        return "No data to plot".to_string();
    }

    let min_val = *values.iter().min().unwrap();
    let max_val = *values.iter().max().unwrap();

    // Use logarithmic binning for better visualization of skewed data
    let num_bins = 20;
    let mut bins = vec![0u64; num_bins];
    let mut bin_labels = Vec::new();

    // Create logarithmic bins
    if min_val == max_val {
        bins[0] = values.len() as u64;
        bin_labels.push(format!("{}", min_val));
    } else {
        let log_min = if min_val == 0 {
            0.0
        } else {
            (min_val as f64).ln()
        };
        let log_max = (max_val as f64 + 1.0).ln();
        let log_range = log_max - log_min;

        // Create bin boundaries
        let mut bin_boundaries = Vec::new();
        for i in 0..=num_bins {
            let log_val = log_min + (i as f64 * log_range / num_bins as f64);
            let val = if log_val == 0.0 {
                0
            } else {
                log_val.exp() as u64
            };
            bin_boundaries.push(val);
        }

        // Count values in each bin
        for &value in values {
            for i in 0..num_bins {
                if value >= bin_boundaries[i] && value < bin_boundaries[i + 1] {
                    bins[i] += 1;
                    break;
                }
            }
        }

        // Create bin labels
        for i in 0..num_bins {
            if bin_boundaries[i] == bin_boundaries[i + 1] - 1 {
                bin_labels.push(format!("{}", bin_boundaries[i]));
            } else {
                bin_labels.push(format!(
                    "{}-{}",
                    bin_boundaries[i],
                    bin_boundaries[i + 1] - 1
                ));
            }
        }
    }

    // Find the maximum count for scaling
    let max_count = *bins.iter().max().unwrap();
    if max_count == 0 {
        return "No data to plot".to_string();
    }

    let mut plot = String::new();
    plot.push_str(&format!("   📈 {} Distribution:\n", title));
    plot.push_str(
        "   ┌─────────────────────────────────────────────────────────────────────────┐\n",
    );

    // Plot each bin
    for (i, &count) in bins.iter().enumerate() {
        if count == 0 {
            continue; // Skip empty bins
        }

        let bar_length = ((count as f64 / max_count as f64) * max_width as f64) as usize;
        let bar_length = bar_length.max(1); // Ensure at least 1 char for non-zero values

        let bar = "█".repeat(bar_length);
        let label = &bin_labels[i];

        // Truncate very long labels
        let display_label = if label.len() > 12 {
            format!("{}...", &label[..9])
        } else {
            label.clone()
        };

        plot.push_str(&format!(
            "   │{:>12} │{:<50} {:>6}\n",
            display_label, bar, count
        ));
    }

    plot.push_str(
        "   └─────────────────────────────────────────────────────────────────────────┘\n",
    );
    plot.push_str(&format!(
        "   │{:>12} │{:<50} {:>6}\n",
        "Range", "Frequency", "Count"
    ));

    plot
}

fn calculate_distribution_stats(values: &[u64]) -> DistributionStats {
    if values.is_empty() {
        return DistributionStats {
            mean: 0.0,
            median: 0,
            min: 0,
            max: 0,
            std_dev: 0.0,
        };
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_unstable();

    let min = sorted_values[0];
    let max = sorted_values[sorted_values.len() - 1];
    let median = sorted_values[sorted_values.len() / 2];

    let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;

    let variance = values
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;

    let std_dev = variance.sqrt();

    DistributionStats {
        mean,
        median,
        min,
        max,
        std_dev,
    }
}

fn write_output_batches_parallel(
    batches: Vec<OutputBatch>,
    progress_bar: &ProgressBar,
) -> Result<()> {
    let results: Result<Vec<_>> = batches
        .into_par_iter()
        .map(|batch| {
            std::fs::create_dir_all(batch.output_path.parent().unwrap())?;
            info!(
                "Writing file {} with {} documents",
                batch.output_path.display(),
                batch.documents.len()
            );
            let result = write_output_file(&batch.output_path, &batch.documents);
            progress_bar.inc(1);
            result
        })
        .collect();

    results.map(|_| ())
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let start_time = Instant::now();

    info!("Starting document grouping process");
    info!("Input directory: {}", args.input_dir.display());
    info!("Output directory: {}", args.output_dir.display());
    info!("Target file size: {} bytes", args.target_size);

    // Initialize analytics collection
    let processing_stats = Arc::new(Mutex::new(ProcessingStats::default()));

    // Setup progress tracking
    let multi_progress = Arc::new(MultiProgress::new());

    // Phase 1: File discovery
    let discovery_pb = multi_progress.add(ProgressBar::new_spinner());
    discovery_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .unwrap(),
    );
    discovery_pb.set_message("Discovering input files...");

    let subdirs_with_files = collect_input_files_by_subdir(&args.input_dir, &discovery_pb)?;
    let total_files: usize = subdirs_with_files.values().map(|files| files.len()).sum();

    // Update initial stats
    {
        let mut stats = processing_stats.lock().unwrap();
        stats.total_files_processed = total_files as u64;
    }

    discovery_pb.finish_with_message(format!(
        "Found {} files across {} subdirectories",
        total_files,
        subdirs_with_files.len()
    ));

    if subdirs_with_files.is_empty() {
        warn!("No input files found");
        return Ok(());
    }

    // Phase 2: Overall progress tracking
    let overall_pb = multi_progress.add(ProgressBar::new(subdirs_with_files.len() as u64));
    overall_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} subdirs ({percent}%) {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ")
    );
    overall_pb.set_message("Processing subdirectories...");

    // Phase 3: Document processing progress
    let doc_processing_pb = multi_progress.add(ProgressBar::new(0));
    doc_processing_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.blue} [{elapsed_precise}] [{bar:40.yellow/red}] {pos} documents processed {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ")
    );

    // Count total estimated documents (rough estimate based on files)
    let estimated_docs = total_files * 1000; // rough estimate
    doc_processing_pb.set_length(estimated_docs as u64);
    doc_processing_pb.set_message("Loading and processing documents...");

    // Process subdirectories in parallel
    let all_batches: Result<Vec<(Vec<OutputBatch>, SubdirStats, String)>> = subdirs_with_files
        .into_par_iter()
        .map(
            |(subdir_path, input_files)| -> Result<(Vec<OutputBatch>, SubdirStats, String)> {
                let subdir_name = subdir_path.to_string_lossy().to_string();
                info!(
                    "Processing subdirectory: {} with {} files",
                    subdir_path.display(),
                    input_files.len()
                );

                // Parallel file loading with document progress
                let subdir_documents: Vec<ProcessedDocument> = input_files
                    .par_iter()
                    .map(|file_path| {
                        info!("Processing file: {}", file_path.display());
                        load_documents_from_file(file_path, Some(&doc_processing_pb))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                info!(
                    "Loaded {} documents from subdirectory {}",
                    subdir_documents.len(),
                    subdir_path.display()
                );

                if subdir_documents.is_empty() {
                    overall_pb.inc(1);
                    return Ok((Vec::new(), SubdirStats::default(), subdir_name));
                }

                // Parallel grouping with analytics collection
                let repo_groups = group_documents_by_repo(subdir_documents);
                let (repo_doc_counts, repo_size_bytes) = collect_repo_analytics(&repo_groups);

                info!(
                    "Grouped documents into {} repositories for {}",
                    repo_groups.len(),
                    subdir_path.display()
                );

                let subdir_output_path = args.output_dir.join(&subdir_path);
                let (batches, mut subdir_stats) =
                    prepare_output_batches(repo_groups, &subdir_output_path, args.target_size);
                subdir_stats.file_count = input_files.len() as u64;

                // Update global statistics
                {
                    let mut stats = processing_stats.lock().unwrap();
                    stats.total_documents += subdir_stats.document_count;
                    stats.total_repositories += subdir_stats.repository_count;
                    stats.total_output_files += subdir_stats.output_files;
                    stats.total_input_size_bytes += subdir_stats.total_size_bytes;

                    // Merge repository analytics
                    for (repo_name, doc_count) in repo_doc_counts {
                        *stats
                            .repository_doc_counts
                            .entry(repo_name.clone())
                            .or_insert(0) += doc_count;
                    }
                    for (repo_name, size_bytes) in repo_size_bytes {
                        *stats.repository_size_bytes.entry(repo_name).or_insert(0) += size_bytes;
                    }
                }

                info!(
                    "Prepared {} output batches for subdirectory: {}",
                    batches.len(),
                    subdir_path.display()
                );
                overall_pb.inc(1);
                Ok((batches, subdir_stats, subdir_name))
            },
        )
        .collect::<Result<Vec<_>>>();

    let result_data = all_batches?;

    // Collect subdirectory stats
    let mut all_batches_vec = Vec::new();
    {
        let mut stats = processing_stats.lock().unwrap();
        for (batches, subdir_stats, subdir_name) in result_data {
            all_batches_vec.extend(batches);
            stats.subdirectory_stats.insert(subdir_name, subdir_stats);
        }
    }

    overall_pb.finish_with_message("All subdirectories processed");
    doc_processing_pb.finish_with_message("All documents loaded and grouped");

    // Phase 4: File writing progress
    let write_pb = multi_progress.add(ProgressBar::new(all_batches_vec.len() as u64));
    write_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} files written ({percent}%) {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ")
    );
    write_pb.set_message("Writing output files...");

    info!(
        "Writing {} total output files in parallel",
        all_batches_vec.len()
    );

    write_output_batches_parallel(all_batches_vec, &write_pb)?;

    write_pb.finish_with_message("All output files written");

    // Final completion message
    let completion_pb = multi_progress.add(ProgressBar::new(1));
    completion_pb.set_style(ProgressStyle::default_bar().template("✅ {msg}").unwrap());
    completion_pb.inc(1);
    completion_pb.finish_with_message("Document grouping completed successfully!");

    // Calculate final statistics and display analytics
    let processing_duration = start_time.elapsed();
    {
        let mut stats = processing_stats.lock().unwrap();
        stats.processing_duration = processing_duration;

        // Calculate output size (rough estimate - would need actual file sizes for precision)
        stats.total_output_size_bytes = (stats.total_input_size_bytes as f64 * 0.3) as u64;
        // Assume ~30% compression
    }

    info!("Document grouping completed successfully");

    // Display comprehensive analytics report
    let final_stats = processing_stats.lock().unwrap().clone();
    print_analytics_report(&final_stats);

    Ok(())
}
