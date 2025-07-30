use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicUsize, atomic::Ordering, Mutex};
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "filter-low-qual-repos")]
#[command(about = "Aggregates document scores by repository and bins them by score ranges")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Aggregate document scores by repository across programming language directories
    Aggregate {
        /// Source directory containing programming language subdirectories
        #[arg(short, long)]
        source_dir: PathBuf,
        
        /// Output report file path
        #[arg(short, long, default_value = "repo_scores_report.json")]
        output: PathBuf,
    },
    /// Bin repositories by average score using reservoir sampling
    Bin {
        /// Source directory containing programming language subdirectories (only used if --build-report is specified)
        #[arg(short, long)]
        source_dir: Option<PathBuf>,
        
        /// Path to existing score report file
        #[arg(short, long, default_value = "repo_scores_report.json")]
        report_path: PathBuf,
        
        /// Force rebuild the score report from source directory
        #[arg(long)]
        build_report: bool,
        
        /// Number of bins to create
        #[arg(short, long, default_value = "10")]
        num_bins: usize,
        
        /// Sample size per bin using reservoir sampling
        #[arg(long, default_value = "100")]
        sample_size: usize,
        
        /// Output report file path
        #[arg(short, long, default_value = "repo_bins_report.json")]
        output: PathBuf,
    },
    /// Filter documents based on repository bin criteria
    Filter {
        /// Source directory containing programming language subdirectories with *.jsonl.zst files
        #[arg(short, long)]
        source_dir: PathBuf,
        
        /// Path to score report file
        #[arg(short, long)]
        report_path: PathBuf,
        
        /// Path to bin report file
        #[arg(short, long)]
        bin_path: PathBuf,
        
        /// Target bin numbers to include (comma-separated, 1-indexed)
        #[arg(short, long, value_delimiter = ',')]
        target_bins: Vec<usize>,
        
        /// Specific language to filter (optional, filters all languages if not specified)
        #[arg(short, long)]
        language: Option<String>,
        
        /// Output directory for filtered *.jsonl.zst files
        #[arg(short, long)]
        output_dir: PathBuf,
        
        /// Maximum file size in MB before splitting
        #[arg(long, default_value = "50")]
        max_file_size_mb: usize,
    },
}

#[derive(Deserialize)]
struct Document {
    #[serde(default)]
    metadata: Metadata,
}

#[derive(Deserialize, Default)]
struct Metadata {
    #[serde(default)]
    repo_name: Option<String>,
    #[serde(default)]
    score: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct RepoStats {
    document_count: usize,
    total_score: f64,
    average_score: f64,
    min_score: f64,
    max_score: f64,
}

#[derive(Serialize, Deserialize)]
struct LanguageReport {
    repo_stats: HashMap<String, RepoStats>,
    total_documents: usize,
    total_repositories: usize,
}

#[derive(Serialize, Deserialize)]
struct Report {
    languages: HashMap<String, LanguageReport>,
    summary: Summary,
}

#[derive(Serialize, Deserialize)]
struct Summary {
    total_languages: usize,
    total_repositories: usize,
    total_documents: usize,
}

#[derive(Serialize, Deserialize)]
struct BinReport {
    language_bins: HashMap<String, LanguageBinReport>,
    summary: BinSummary,
}

#[derive(Serialize, Deserialize)]
struct LanguageBinReport {
    language: String,
    bins: Vec<ScoreBin>,
    summary: LanguageBinSummary,
}

#[derive(Serialize, Deserialize)]
struct ScoreBin {
    min_score: f64,
    max_score: f64,
    sample_repos: Vec<BinRepo>,
    total_repos_in_range: usize,
}

#[derive(Serialize, Deserialize)]
struct BinRepo {
    repo_name: String,
    language: String,
    average_score: f64,
    document_count: usize,
}

#[derive(Serialize, Deserialize)]
struct BinSummary {
    total_languages: usize,
    total_repositories: usize,
    total_documents: usize,
    num_bins: usize,
    sample_size_per_bin: usize,
}

#[derive(Serialize, Deserialize)]
struct LanguageBinSummary {
    language: String,
    total_repositories: usize,
    total_documents: usize,
    num_bins: usize,
    sample_size_per_bin: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    match args.command {
        Commands::Aggregate { source_dir, output } => {
            aggregate_scores(source_dir, output)
        }
        Commands::Bin { source_dir, report_path, build_report, num_bins, sample_size, output } => {
            bin_repositories(source_dir, report_path, build_report, num_bins, sample_size, output)
        }
        Commands::Filter { source_dir, report_path, bin_path, target_bins, language, output_dir, max_file_size_mb } => {
            filter_documents(source_dir, report_path, bin_path, target_bins, language, output_dir, max_file_size_mb)
        }
    }
}

fn aggregate_scores(source_dir: PathBuf, output: PathBuf) -> Result<()> {
    let start_time = Instant::now();
    println!("Starting aggregation process...");
    
    if !source_dir.exists() {
        anyhow::bail!("Source directory does not exist: {}", source_dir.display());
    }
    
    let mut report = Report {
        languages: HashMap::new(),
        summary: Summary {
            total_languages: 0,
            total_repositories: 0,
            total_documents: 0,
        },
    };
    
    // Check if source_dir contains *.jsonl.zst files directly (single language mode)
    let has_jsonl_files = std::fs::read_dir(&source_dir)?
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let path = entry.path();
            path.is_file() && 
            path.extension().and_then(|s| s.to_str()) == Some("zst") &&
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.ends_with(".jsonl"))
                .unwrap_or(false)
        });
    
    if has_jsonl_files {
        // Single language directory mode
        let language_name = source_dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        println!("[1/1] Processing single language directory: {}", language_name);
        
        match process_language_directory(&source_dir) {
            Ok(language_report) => {
                println!("  ✓ Completed {} - {} repos, {} docs", 
                         language_name, language_report.total_repositories, language_report.total_documents);
                report.languages.insert(language_name, language_report);
            }
            Err(e) => {
                eprintln!("  ✗ Error processing {}: {}", language_name, e);
            }
        }
    } else {
        // Multi-language directory mode - count directories first for progress
        let language_dirs: Vec<_> = std::fs::read_dir(&source_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_dir())
            .collect();
        
        let total_langs = language_dirs.len();
        println!("Found {} language directories to process", total_langs);
        
        for (idx, entry) in language_dirs.into_iter().enumerate() {
            let path = entry.path();
            let language_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            
            println!("[{}/{}] Processing language directory: {}", idx + 1, total_langs, language_name);
            
            match process_language_directory(&path) {
                Ok(language_report) => {
                    println!("  ✓ Completed {} - {} repos, {} docs", 
                             language_name, language_report.total_repositories, language_report.total_documents);
                    report.languages.insert(language_name, language_report);
                }
                Err(e) => {
                    eprintln!("  ✗ Error processing {}: {}", language_name, e);
                }
            }
        }
    }
    
    // Calculate summary statistics
    report.summary.total_languages = report.languages.len();
    report.summary.total_repositories = report.languages.values()
        .map(|lang| lang.total_repositories)
        .sum();
    report.summary.total_documents = report.languages.values()
        .map(|lang| lang.total_documents)
        .sum();
    
    // Write report to file
    let mut output_file = File::create(&output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;
    
    let json = serde_json::to_string_pretty(&report)?;
    output_file.write_all(json.as_bytes())?;
    
    let elapsed = start_time.elapsed();
    println!("Report written to: {}", output.display());
    println!("Summary:");
    println!("  Languages: {}", report.summary.total_languages);
    println!("  Repositories: {}", report.summary.total_repositories);
    println!("  Documents: {}", report.summary.total_documents);
    println!("  Completed in: {:.2}s", elapsed.as_secs_f64());
    
    Ok(())
}

fn bin_repositories(
    source_dir: Option<PathBuf>, 
    report_path: PathBuf, 
    build_report: bool, 
    num_bins: usize, 
    sample_size: usize, 
    output: PathBuf
) -> Result<()> {
    let start_time = Instant::now();
    println!("Starting repository binning process...");
    
    let report = if build_report {
        // Force rebuild the report
        let source_dir = source_dir.ok_or_else(|| {
            anyhow::anyhow!("--source-dir is required when using --build-report")
        })?;
        
        if !source_dir.exists() {
            anyhow::bail!("Source directory does not exist: {}", source_dir.display());
        }
        
        println!("Building new score report from source directory...");
        build_score_report(source_dir, report_path.clone())?;
        
        // Read the newly created report
        load_score_report(&report_path)?
    } else {
        // Try to load existing report
        if report_path.exists() {
            println!("Using existing score report: {}", report_path.display());
            load_score_report(&report_path)?
        } else {
            anyhow::bail!(
                "Score report file does not exist: {}. Use --build-report flag to create it from source data.",
                report_path.display()
            );
        }
    };

    // Create bins per language
    println!("Creating language-specific bins...");
    let mut language_bins = HashMap::new();
    let mut total_repos = 0;
    let mut total_docs = 0;
    
    for (language, language_report) in &report.languages {
        println!("Processing language: {}", language);
        
        // Extract repositories for this language
        let mut language_repos: Vec<(String, f64, usize)> = Vec::new(); // (repo_name, avg_score, doc_count)
        for (repo_name, repo_stats) in &language_report.repo_stats {
            language_repos.push((
                repo_name.clone(),
                repo_stats.average_score,
                repo_stats.document_count
            ));
            total_docs += repo_stats.document_count;
        }
        
        if language_repos.is_empty() {
            println!("  No repositories found for {}, skipping", language);
            continue;
        }
        
        total_repos += language_repos.len();
        
        // Sort repositories by average score for this language
        language_repos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let min_score = language_repos.first().map(|r| r.1).unwrap_or(0.0);
        let max_score = language_repos.last().map(|r| r.1).unwrap_or(1.0);
        
        println!("  {} repositories, score range: {:.6} to {:.6}", 
                 language_repos.len(), min_score, max_score);
        
        // Create bins for this language
        let lang_bins = create_language_bins(&language_repos, num_bins, sample_size, min_score, max_score)?;
        
        let lang_bin_report = LanguageBinReport {
            language: language.clone(),
            bins: lang_bins,
            summary: LanguageBinSummary {
                language: language.clone(),
                total_repositories: language_repos.len(),
                total_documents: language_report.total_documents,
                num_bins,
                sample_size_per_bin: sample_size,
            },
        };
        
        language_bins.insert(language.clone(), lang_bin_report);
    }

    let bin_report = BinReport {
        language_bins,
        summary: BinSummary {
            total_languages: report.summary.total_languages,
            total_repositories: total_repos,
            total_documents: total_docs,
            num_bins,
            sample_size_per_bin: sample_size,
        },
    };

    // Write report to file
    let mut output_file = File::create(&output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;
    
    let json = serde_json::to_string_pretty(&bin_report)?;
    output_file.write_all(json.as_bytes())?;
    
    let elapsed = start_time.elapsed();
    println!("Bin report written to: {}", output.display());
    println!("Summary:");
    println!("  Languages: {}", bin_report.summary.total_languages);
    println!("  Repositories: {}", bin_report.summary.total_repositories);
    println!("  Documents: {}", bin_report.summary.total_documents);
    println!("  Bins: {}", bin_report.summary.num_bins);
    println!("  Sample size per bin: {}", bin_report.summary.sample_size_per_bin);
    println!("  Completed in: {:.2}s", elapsed.as_secs_f64());
    
    Ok(())
}

fn filter_documents(
    source_dir: PathBuf,
    report_path: PathBuf,
    bin_path: PathBuf,
    target_bins: Vec<usize>,
    target_language: Option<String>,
    output_dir: PathBuf,
    max_file_size_mb: usize,
) -> Result<()> {
    let start_time = Instant::now();
    println!("Starting document filtering process...");
    
    // Validate inputs
    if !source_dir.exists() {
        anyhow::bail!("Source directory does not exist: {}", source_dir.display());
    }
    if !report_path.exists() {
        anyhow::bail!("Report file does not exist: {}", report_path.display());
    }
    if !bin_path.exists() {
        anyhow::bail!("Bin file does not exist: {}", bin_path.display());
    }
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;
    
    // Load bin report to get target repositories
    println!("Loading bin report...");
    let bin_report = load_bin_report(&bin_path)?;
    
    // Extract target repositories from specified bins using score ranges (streaming)
    let target_repos = extract_target_repos_streaming_by_language(&bin_report, &report_path, &target_bins, &target_language)?;
    println!("Found {} target repositories across {} bins", target_repos.len(), target_bins.len());
    
    // Get all source files with their language directories
    println!("Scanning source files...");
    let source_files = collect_source_files_with_language(&source_dir)?;
    println!("Found {} source files across {} languages to process", 
             source_files.len(), 
             source_files.iter().map(|(lang, _)| lang).collect::<std::collections::HashSet<_>>().len());
    
    // Setup progress bar
    let pb = ProgressBar::new(source_files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%) {msg}")
        .unwrap()
        .progress_chars("#>-"));
    
    // Process files and collect matching documents
    let max_file_size_bytes = max_file_size_mb * 1024 * 1024;
    let output_manager = Mutex::new(OutputManager::new(output_dir.clone(), max_file_size_bytes)?);
    let processed_docs = AtomicUsize::new(0);
    let filtered_docs = AtomicUsize::new(0);
    
    source_files.par_iter().for_each(|(language, file_path)| {
        match process_source_file_with_language(file_path, language, &target_repos, &output_manager, &processed_docs, &filtered_docs) {
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error processing {}: {}", file_path.display(), e);
            }
        }
        pb.inc(1);
    });
    
    pb.finish_with_message("Processing complete");
    
    // Finalize output files
    let mut output_mgr = output_manager.into_inner().unwrap();
    output_mgr.finalize()?;
    
    let elapsed = start_time.elapsed();
    let total_processed = processed_docs.load(Ordering::Relaxed);
    let total_retained = filtered_docs.load(Ordering::Relaxed);
    let total_excluded = total_processed - total_retained;
    
    println!("Filter completed in {:.2}s", elapsed.as_secs_f64());
    println!("Summary:");
    println!("  Documents processed: {}", total_processed);
    println!("  Documents retained: {}", total_retained);
    println!("  Documents excluded: {}", total_excluded);
    println!("  Retention rate: {:.2}%", if total_processed > 0 { 
        (total_retained as f64 / total_processed as f64) * 100.0 
    } else { 0.0 });
    println!("  Output files: {}", output_mgr.file_count());
    println!("  Output directory: {}", output_dir.display());
    
    Ok(())
}

fn create_language_bins(
    language_repos: &[(String, f64, usize)], // (repo_name, avg_score, doc_count)
    num_bins: usize,
    sample_size: usize,
    min_score: f64,
    max_score: f64,
) -> Result<Vec<ScoreBin>> {
    let bin_width = (max_score - min_score) / num_bins as f64;
    
    // Pre-calculate bin ranges
    let bin_ranges: Vec<(usize, f64, f64)> = (0..num_bins)
        .map(|i| {
            let bin_min = min_score + i as f64 * bin_width;
            let bin_max = if i == num_bins - 1 { max_score } else { min_score + (i + 1) as f64 * bin_width };
            (i, bin_min, bin_max)
        })
        .collect();
    
    // Process bins in parallel
    let bins: Vec<ScoreBin> = bin_ranges
        .par_iter()
        .map(|(i, bin_min, bin_max)| {
            // Find all repos in this bin
            let repos_in_bin: Vec<_> = language_repos
                .iter()
                .filter(|(_, score, _)| *score >= *bin_min && *score <= *bin_max)
                .collect();
            
            let total_repos_in_range = repos_in_bin.len();
            
            // Apply reservoir sampling with thread-local RNG
            let sample_repos: Vec<BinRepo> = if repos_in_bin.len() <= sample_size {
                // If we have fewer repos than sample size, take all
                repos_in_bin.into_iter().map(|(repo_name, avg_score, doc_count)| {
                    BinRepo {
                        repo_name: repo_name.clone(),
                        language: "".to_string(), // Will be set by caller
                        average_score: *avg_score,
                        document_count: *doc_count,
                    }
                }).collect()
            } else {
                // Reservoir sampling with thread-local RNG
                let mut rng = thread_rng();
                let sampled: Vec<_> = repos_in_bin.choose_multiple(&mut rng, sample_size).collect();
                sampled.into_iter().map(|(repo_name, avg_score, doc_count)| {
                    BinRepo {
                        repo_name: repo_name.clone(),
                        language: "".to_string(), // Will be set by caller
                        average_score: *avg_score,
                        document_count: *doc_count,
                    }
                }).collect()
            };
            
            println!("    Bin {}: [{:.6}, {:.6}] - {} repos total, {} sampled", 
                     i + 1, bin_min, bin_max, total_repos_in_range, sample_repos.len());
            
            ScoreBin {
                min_score: *bin_min,
                max_score: *bin_max,
                sample_repos,
                total_repos_in_range,
            }
        })
        .collect();
    
    Ok(bins)
}

fn process_language_directory(dir_path: &Path) -> Result<LanguageReport> {
    // Collect all *.jsonl.zst files first
    let files: Vec<PathBuf> = WalkDir::new(dir_path)
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            
            if path.is_file() && 
               path.extension().and_then(|s| s.to_str()) == Some("zst") &&
               path.file_stem()
                   .and_then(|s| s.to_str())
                   .map(|s| s.ends_with(".jsonl"))
                   .unwrap_or(false) {
                Some(path.to_path_buf())
            } else {
                None
            }
        })
        .collect();
    
    println!("  Found {} files to process", files.len());
    
    // Use Mutex to safely collect results from parallel processing
    let repo_scores = Mutex::new(HashMap::<String, Vec<f64>>::new());
    let total_documents = Mutex::new(0usize);
    let processed_files = AtomicUsize::new(0);
    
    // Process files in parallel
    files.par_iter().for_each(|file_path| {
        let current = processed_files.fetch_add(1, Ordering::Relaxed) + 1;
        println!("  [{}/{}] Processing file: {}", current, files.len(), file_path.display());
        
        match process_jsonl_zst_file(file_path) {
            Ok((docs, repo_data)) => {
                // Update total documents count
                {
                    let mut total_docs = total_documents.lock().unwrap();
                    *total_docs += docs;
                }
                
                // Update repository scores
                {
                    let mut repo_scores_map = repo_scores.lock().unwrap();
                    for (repo_name, scores) in repo_data {
                        repo_scores_map.entry(repo_name)
                            .or_insert_with(Vec::new)
                            .extend(scores);
                    }
                }
            }
            Err(e) => {
                eprintln!("    Warning: Failed to process {}: {}", file_path.display(), e);
            }
        }
    });
    
    // Extract results from Mutex
    let repo_scores = repo_scores.into_inner().unwrap();
    let total_documents = total_documents.into_inner().unwrap();
    
    // Calculate statistics for each repository
    let mut repo_stats = HashMap::new();
    for (repo_name, scores) in repo_scores {
        if !scores.is_empty() {
            let total_score: f64 = scores.iter().sum();
            let min_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let average_score = total_score / scores.len() as f64;
            
            repo_stats.insert(repo_name, RepoStats {
                document_count: scores.len(),
                total_score,
                average_score,
                min_score,
                max_score,
            });
        }
    }
    
    Ok(LanguageReport {
        total_repositories: repo_stats.len(),
        total_documents,
        repo_stats,
    })
}

fn process_jsonl_zst_file(file_path: &Path) -> Result<(usize, HashMap<String, Vec<f64>>)> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    
    let decoder = zstd::Decoder::new(file)
        .with_context(|| format!("Failed to create zstd decoder for: {}", file_path.display()))?;
    
    let reader = BufReader::new(decoder);
    let mut repo_scores: HashMap<String, Vec<f64>> = HashMap::new();
    let mut document_count = 0;
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!("Failed to read line {} from: {}", line_num + 1, file_path.display())
        })?;
        
        if line.trim().is_empty() {
            continue;
        }
        
        match serde_json::from_str::<Document>(&line) {
            Ok(doc) => {
                document_count += 1;
                
                if let (Some(repo_name), Some(score)) = (doc.metadata.repo_name, doc.metadata.score) {
                    repo_scores.entry(repo_name)
                        .or_insert_with(Vec::new)
                        .push(score);
                }
            }
            Err(e) => {
                eprintln!("    Warning: Failed to parse JSON on line {}: {}", line_num + 1, e);
            }
        }
    }
    
    Ok((document_count, repo_scores))
}

fn build_score_report(source_dir: PathBuf, output_path: PathBuf) -> Result<()> {
    aggregate_scores(source_dir, output_path)
}

fn load_score_report(report_path: &Path) -> Result<Report> {
    let file = File::open(report_path)
        .with_context(|| format!("Failed to open score report: {}", report_path.display()))?;
    
    let report: Report = serde_json::from_reader(file)
        .with_context(|| format!("Failed to parse score report: {}", report_path.display()))?;
    
    Ok(report)
}

fn load_bin_report(bin_path: &Path) -> Result<BinReport> {
    let file = File::open(bin_path)
        .with_context(|| format!("Failed to open bin report: {}", bin_path.display()))?;
    
    let report: BinReport = serde_json::from_reader(file)
        .with_context(|| format!("Failed to parse bin report: {}", bin_path.display()))?;
    
    Ok(report)
}

fn extract_target_repos_streaming_by_language(
    bin_report: &BinReport,
    report_path: &Path,
    target_bins: &[usize],
    target_language: &Option<String>
) -> Result<HashSet<String>> {
    println!("Extracting target repositories from score report (streaming)...");
    
    // Get score ranges for target bins across all languages (or specific language)
    let mut language_score_ranges: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    
    let languages_to_process: Vec<String> = if let Some(lang) = target_language {
        vec![lang.clone()]
    } else {
        bin_report.language_bins.keys().cloned().collect()
    };
    
    for language in &languages_to_process {
        if let Some(lang_bin_report) = bin_report.language_bins.get(language) {
            let mut score_ranges = Vec::new();
            
            for &bin_idx in target_bins {
                if bin_idx == 0 || bin_idx > lang_bin_report.bins.len() {
                    anyhow::bail!("Invalid bin number: {} for language {}. Valid range: 1-{}", 
                                 bin_idx, language, lang_bin_report.bins.len());
                }
                
                let bin = &lang_bin_report.bins[bin_idx - 1]; // Convert to 0-indexed
                score_ranges.push((bin.min_score, bin.max_score));
                println!("Target bin {} for {}: score range [{:.6}, {:.6}]", 
                         bin_idx, language, bin.min_score, bin.max_score);
            }
            
            language_score_ranges.insert(language.clone(), score_ranges);
        } else {
            if target_language.is_some() {
                anyhow::bail!("Language '{}' not found in bin report", language);
            }
            println!("Warning: Language '{}' not found in bin report, skipping", language);
        }
    }
    
    let file = File::open(report_path)
        .with_context(|| format!("Failed to open score report: {}", report_path.display()))?;
    
    parse_score_report_for_language_ranges(file, &language_score_ranges)
}

fn parse_score_report_for_language_ranges(
    file: File,
    language_score_ranges: &HashMap<String, Vec<(f64, f64)>>
) -> Result<HashSet<String>> {
    
    let mut target_repos = HashSet::new();
    let reader = BufReader::new(file);
    
    // Use serde_json's streaming API to parse the structure
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    
    // Parse the JSON structure piece by piece
    let value: serde_json::Value = serde_json::Value::deserialize(&mut deserializer)?;
    
    if let Some(languages) = value.get("languages").and_then(|v| v.as_object()) {
        let mut processed_repos = 0;
        let total_languages = languages.len();
        
        for (lang_idx, (language_name, language_data)) in languages.iter().enumerate() {
            if lang_idx % 10 == 0 {
                println!("  Processing language {}/{} - found {} repos so far", 
                         lang_idx + 1, total_languages, target_repos.len());
            }
            
            // Check if this language has target score ranges
            if let Some(score_ranges) = language_score_ranges.get(language_name) {
                if let Some(repo_stats) = language_data.get("repo_stats").and_then(|v| v.as_object()) {
                    for (repo_name, repo_data) in repo_stats {
                        processed_repos += 1;
                        
                        if let Some(avg_score) = repo_data.get("average_score").and_then(|v| v.as_f64()) {
                            // Check if repository falls within any target score range for this language
                            for &(min_score, max_score) in score_ranges {
                                if avg_score >= min_score && avg_score <= max_score {
                                    target_repos.insert(repo_name.clone());
                                    break;
                                }
                            }
                        }
                        
                        // Progress update every 100k repos
                        if processed_repos % 100000 == 0 {
                            println!("  Processed {} repositories, found {} matches", 
                                     processed_repos, target_repos.len());
                        }
                    }
                }
            }
        }
        
        println!("  Completed processing {} repositories total", processed_repos);
    }
    
    Ok(target_repos)
}

fn collect_source_files_with_language(source_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut source_files = Vec::new();
    
    // Check if source_dir contains *.jsonl.zst files directly (single language mode)
    let has_jsonl_files = std::fs::read_dir(source_dir)?
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let path = entry.path();
            path.is_file() && 
            path.extension().and_then(|s| s.to_str()) == Some("zst") &&
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.ends_with(".jsonl"))
                .unwrap_or(false)
        });
    
    if has_jsonl_files {
        // Single language directory mode
        let language_name = source_dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        for entry in WalkDir::new(source_dir) {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && 
               path.extension().and_then(|s| s.to_str()) == Some("zst") &&
               path.file_stem()
                   .and_then(|s| s.to_str())
                   .map(|s| s.ends_with(".jsonl"))
                   .unwrap_or(false) {
                source_files.push((language_name.clone(), path.to_path_buf()));
            }
        }
    } else {
        // Multi-language directory mode
        for entry in std::fs::read_dir(source_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let language_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                // Collect files from this language directory
                for file_entry in WalkDir::new(&path) {
                    let file_entry = file_entry?;
                    let file_path = file_entry.path();
                    
                    if file_path.is_file() && 
                       file_path.extension().and_then(|s| s.to_str()) == Some("zst") &&
                       file_path.file_stem()
                           .and_then(|s| s.to_str())
                           .map(|s| s.ends_with(".jsonl"))
                           .unwrap_or(false) {
                        source_files.push((language_name.clone(), file_path.to_path_buf()));
                    }
                }
            }
        }
    }
    
    Ok(source_files)
}

fn process_source_file_with_language(
    file_path: &Path,
    language: &str,
    target_repos: &HashSet<String>,
    output_manager: &Mutex<OutputManager>,
    processed_docs: &AtomicUsize,
    filtered_docs: &AtomicUsize,
) -> Result<()> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    
    let decoder = zstd::Decoder::new(file)
        .with_context(|| format!("Failed to create zstd decoder for: {}", file_path.display()))?;
    
    let reader = BufReader::new(decoder);
    let mut repo_docs: HashMap<String, Vec<String>> = HashMap::new();
    
    // Group documents by repository
    for line in reader.lines() {
        let line = line?;
        processed_docs.fetch_add(1, Ordering::Relaxed);
        
        if line.trim().is_empty() {
            continue;
        }
        
        // Parse just enough to get repo_name
        if let Ok(doc) = serde_json::from_str::<Document>(&line) {
            if let Some(repo_name) = doc.metadata.repo_name {
                if target_repos.contains(&repo_name) {
                    repo_docs.entry(repo_name)
                        .or_insert_with(Vec::new)
                        .push(line);
                    filtered_docs.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
    
    // Write documents to output, grouped by repository
    if !repo_docs.is_empty() {
        let mut output_mgr = output_manager.lock().unwrap();
        output_mgr.write_repo_documents_for_language(language, repo_docs)?;
    }
    
    Ok(())
}

struct OutputManager {
    output_dir: PathBuf,
    max_file_size: usize,
    language_managers: HashMap<String, LanguageOutputManager>,
    total_file_count: usize,
}

struct LanguageOutputManager {
    language_dir: PathBuf,
    current_file_idx: usize,
    current_writer: Option<zstd::Encoder<'static, File>>,
    current_size: usize,
    file_count: usize,
}

impl OutputManager {
    fn new(output_dir: PathBuf, max_file_size: usize) -> Result<Self> {
        Ok(Self {
            output_dir,
            max_file_size,
            language_managers: HashMap::new(),
            total_file_count: 0,
        })
    }
    
    fn write_repo_documents_for_language(&mut self, language: &str, repo_docs: HashMap<String, Vec<String>>) -> Result<()> {
        // Get or create language manager
        if !self.language_managers.contains_key(language) {
            let language_dir = self.output_dir.join(language);
            std::fs::create_dir_all(&language_dir)
                .with_context(|| format!("Failed to create language directory: {}", language_dir.display()))?;
            
            self.language_managers.insert(
                language.to_string(),
                LanguageOutputManager {
                    language_dir,
                    current_file_idx: 0,
                    current_writer: None,
                    current_size: 0,
                    file_count: 0,
                }
            );
        }
        
        let lang_manager = self.language_managers.get_mut(language).unwrap();
        lang_manager.write_repo_documents(repo_docs, self.max_file_size)?;
        
        Ok(())
    }
    
    fn finalize(&mut self) -> Result<()> {
        for lang_manager in self.language_managers.values_mut() {
            lang_manager.finalize()?;
            self.total_file_count += lang_manager.file_count;
        }
        Ok(())
    }
    
    fn file_count(&self) -> usize {
        self.total_file_count
    }
}

impl LanguageOutputManager {
    fn write_repo_documents(&mut self, repo_docs: HashMap<String, Vec<String>>, max_file_size: usize) -> Result<()> {
        // Sort repositories for consistent output
        let mut sorted_repos: Vec<_> = repo_docs.into_iter().collect();
        sorted_repos.sort_by(|a, b| a.0.cmp(&b.0));
        
        for (_repo_name, docs) in sorted_repos {
            // Calculate size needed for this repository
            let repo_size: usize = docs.iter().map(|doc| doc.len() + 1).sum(); // +1 for newline
            
            // Check if we need a new file
            if self.current_writer.is_none() || 
               (self.current_size + repo_size > max_file_size && self.current_size > 0) {
                self.rotate_file()?;
            }
            
            // Write all documents for this repository
            let writer = self.current_writer.as_mut().unwrap();
            for doc in docs {
                writer.write_all(doc.as_bytes())?;
                writer.write_all(b"\n")?;
                self.current_size += doc.len() + 1;
            }
        }
        
        Ok(())
    }
    
    fn rotate_file(&mut self) -> Result<()> {
        // Close current file
        if let Some(writer) = self.current_writer.take() {
            writer.finish()?;
        }
        
        // Create new file
        let file_path = self.language_dir.join(format!("{:06}.jsonl.zst", self.current_file_idx));
        let file = File::create(&file_path)
            .with_context(|| format!("Failed to create output file: {}", file_path.display()))?;
        
        let encoder = zstd::Encoder::new(file, 3)?; // Compression level 3
        self.current_writer = Some(encoder);
        self.current_size = 0;
        self.current_file_idx += 1;
        self.file_count += 1;
        
        Ok(())
    }
    
    fn finalize(&mut self) -> Result<()> {
        if let Some(writer) = self.current_writer.take() {
            writer.finish()?;
        }
        Ok(())
    }
}