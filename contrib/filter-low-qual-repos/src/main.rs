use anyhow::{Context, Result};
use clap::Parser;
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
#[command(about = "Aggregates document scores by repository across programming language directories")]
struct Args {
    /// Source directory containing programming language subdirectories
    #[arg(short, long)]
    source_dir: PathBuf,
    
    /// Output report file path
    #[arg(short, long, default_value = "repo_scores_report.json")]
    output: PathBuf,
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

#[derive(Serialize)]
struct RepoStats {
    document_count: usize,
    total_score: f64,
    average_score: f64,
    min_score: f64,
    max_score: f64,
}

#[derive(Serialize)]
struct LanguageReport {
    repo_stats: HashMap<String, RepoStats>,
    total_documents: usize,
    total_repositories: usize,
}

#[derive(Serialize)]
struct Report {
    languages: HashMap<String, LanguageReport>,
    summary: Summary,
}

#[derive(Serialize)]
struct Summary {
    total_languages: usize,
    total_repositories: usize,
    total_documents: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    if !args.source_dir.exists() {
        anyhow::bail!("Source directory does not exist: {}", args.source_dir.display());
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
    let has_jsonl_files = std::fs::read_dir(&args.source_dir)?
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
        let language_name = args.source_dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        println!("Processing single language directory: {}", language_name);
        
        match process_language_directory(&args.source_dir) {
            Ok(language_report) => {
                report.languages.insert(language_name, language_report);
            }
            Err(e) => {
                eprintln!("Error processing {}: {}", language_name, e);
            }
        }
    } else {
        // Multi-language directory mode (original behavior)
        for entry in std::fs::read_dir(&args.source_dir)? {
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
    let mut output_file = File::create(&args.output)
        .with_context(|| format!("Failed to create output file: {}", args.output.display()))?;
    
    let json = serde_json::to_string_pretty(&report)?;
    output_file.write_all(json.as_bytes())?;
    
    println!("Report written to: {}", args.output.display());
    println!("Summary:");
    println!("  Languages: {}", report.summary.total_languages);
    println!("  Repositories: {}", report.summary.total_repositories);
    println!("  Documents: {}", report.summary.total_documents);
    
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