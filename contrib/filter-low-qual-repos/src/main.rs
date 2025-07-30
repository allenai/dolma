use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
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

#[derive(Serialize)]
struct BinReport {
    bins: Vec<ScoreBin>,
    summary: BinSummary,
}

#[derive(Serialize)]
struct ScoreBin {
    min_score: f64,
    max_score: f64,
    sample_repos: Vec<BinRepo>,
    total_repos_in_range: usize,
}

#[derive(Serialize)]
struct BinRepo {
    repo_name: String,
    language: String,
    average_score: f64,
    document_count: usize,
}

#[derive(Serialize)]
struct BinSummary {
    total_languages: usize,
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
    }
}

fn aggregate_scores(source_dir: PathBuf, output: PathBuf) -> Result<()> {
    
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
        
        println!("Processing single language directory: {}", language_name);
        
        match process_language_directory(&source_dir) {
            Ok(language_report) => {
                report.languages.insert(language_name, language_report);
            }
            Err(e) => {
                eprintln!("Error processing {}: {}", language_name, e);
            }
        }
    } else {
        // Multi-language directory mode (original behavior)
        for entry in std::fs::read_dir(&source_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let language_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                println!("Processing language directory: {}", language_name);
                
                match process_language_directory(&path) {
                    Ok(language_report) => {
                        report.languages.insert(language_name, language_report);
                    }
                    Err(e) => {
                        eprintln!("Error processing {}: {}", language_name, e);
                    }
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
    
    println!("Report written to: {}", output.display());
    println!("Summary:");
    println!("  Languages: {}", report.summary.total_languages);
    println!("  Repositories: {}", report.summary.total_repositories);
    println!("  Documents: {}", report.summary.total_documents);
    
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

    // Extract repository data from the report
    let mut all_repos: Vec<(String, String, f64, usize)> = Vec::new(); // (repo_name, language, avg_score, doc_count)
    
    for (language, language_report) in &report.languages {
        for (repo_name, repo_stats) in &language_report.repo_stats {
            all_repos.push((
                repo_name.clone(),
                language.clone(),
                repo_stats.average_score,
                repo_stats.document_count
            ));
        }
    }

    if all_repos.is_empty() {
        anyhow::bail!("No repositories found in the score report");
    }

    // Sort repositories by average score
    all_repos.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    
    let min_score = all_repos.first().map(|r| r.2).unwrap_or(0.0);
    let max_score = all_repos.last().map(|r| r.2).unwrap_or(1.0);
    
    println!("Score range: {:.6} to {:.6}", min_score, max_score);
    println!("Total repositories: {}", all_repos.len());

    // Create bins and apply reservoir sampling
    let mut bins = Vec::new();
    let bin_width = (max_score - min_score) / num_bins as f64;
    
    for i in 0..num_bins {
        let bin_min = min_score + i as f64 * bin_width;
        let bin_max = if i == num_bins - 1 { max_score } else { min_score + (i + 1) as f64 * bin_width };
        
        // Find all repos in this bin
        let repos_in_bin: Vec<_> = all_repos
            .iter()
            .filter(|(_, _, score, _)| *score >= bin_min && *score <= bin_max)
            .collect();
        
        let total_repos_in_range = repos_in_bin.len();
        
        // Apply reservoir sampling
        let sample_repos: Vec<BinRepo> = if repos_in_bin.len() <= sample_size {
            // If we have fewer repos than sample size, take all
            repos_in_bin.into_iter().map(|(repo_name, language, avg_score, doc_count)| {
                BinRepo {
                    repo_name: repo_name.clone(),
                    language: language.clone(),
                    average_score: *avg_score,
                    document_count: *doc_count,
                }
            }).collect()
        } else {
            // Reservoir sampling
            let mut rng = thread_rng();
            let sampled: Vec<_> = repos_in_bin.choose_multiple(&mut rng, sample_size).collect();
            sampled.into_iter().map(|(repo_name, language, avg_score, doc_count)| {
                BinRepo {
                    repo_name: repo_name.clone(),
                    language: language.clone(),
                    average_score: *avg_score,
                    document_count: *doc_count,
                }
            }).collect()
        };
        
        println!("Bin {}: [{:.6}, {:.6}] - {} repos total, {} sampled", 
                 i + 1, bin_min, bin_max, total_repos_in_range, sample_repos.len());
        
        bins.push(ScoreBin {
            min_score: bin_min,
            max_score: bin_max,
            sample_repos,
            total_repos_in_range,
        });
    }

    let bin_report = BinReport {
        bins,
        summary: BinSummary {
            total_languages: report.summary.total_languages,
            total_repositories: all_repos.len(),
            total_documents: report.summary.total_documents,
            num_bins,
            sample_size_per_bin: sample_size,
        },
    };

    // Write report to file
    let mut output_file = File::create(&output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;
    
    let json = serde_json::to_string_pretty(&bin_report)?;
    output_file.write_all(json.as_bytes())?;
    
    println!("Bin report written to: {}", output.display());
    println!("Summary:");
    println!("  Languages: {}", bin_report.summary.total_languages);
    println!("  Repositories: {}", bin_report.summary.total_repositories);
    println!("  Documents: {}", bin_report.summary.total_documents);
    println!("  Bins: {}", bin_report.summary.num_bins);
    println!("  Sample size per bin: {}", bin_report.summary.sample_size_per_bin);
    
    Ok(())
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
    
    // Process files in parallel
    files.par_iter().for_each(|file_path| {
        println!("  Processing file: {}", file_path.display());
        
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