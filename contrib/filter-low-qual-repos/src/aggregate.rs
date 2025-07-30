use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicUsize, atomic::Ordering, Mutex};
use std::time::Instant;
use walkdir::WalkDir;

use crate::types::*;
use crate::utils::*;

pub fn aggregate_scores(source_dir: PathBuf, output: PathBuf) -> Result<()> {
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
            path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("zst")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with(".jsonl"))
                    .unwrap_or(false)
        });

    if has_jsonl_files {
        // Single language directory mode
        let language_name = source_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        println!("[1/1] Processing single language directory: {}", language_name);

        match process_language_directory(&source_dir) {
            Ok(language_report) => {
                println!(
                    "  ✓ Completed {} - {} repos, {} docs",
                    language_name,
                    language_report.total_repositories,
                    language_report.total_documents
                );
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
            let language_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            println!(
                "[{}/{}] Processing language directory: {}",
                idx + 1,
                total_langs,
                language_name
            );

            match process_language_directory(&path) {
                Ok(language_report) => {
                    println!(
                        "  ✓ Completed {} - {} repos, {} docs",
                        language_name,
                        language_report.total_repositories,
                        language_report.total_documents
                    );
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
    report.summary.total_repositories = report
        .languages
        .values()
        .map(|lang| lang.total_repositories)
        .sum();
    report.summary.total_documents = report
        .languages
        .values()
        .map(|lang| lang.total_documents)
        .sum();

    // Write report in Arrow/Parquet format with partitioning
    write_arrow_partitioned_report(&report, &output)?;

    let elapsed = start_time.elapsed();
    println!("Report written to: {}", output.display());
    println!("Summary:");
    println!("  Languages: {}", report.summary.total_languages);
    println!("  Repositories: {}", report.summary.total_repositories);
    println!("  Documents: {}", report.summary.total_documents);
    println!("  Completed in: {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

fn write_arrow_partitioned_report(report: &Report, output_path: &PathBuf) -> Result<()> {
    println!("Writing partitioned Arrow/Parquet report...");

    // Create output directory structure
    let base_dir = if output_path.extension().is_some() {
        output_path.with_extension("")
    } else {
        output_path.clone()
    };

    create_dir_all(&base_dir)
        .with_context(|| format!("Failed to create output directory: {}", base_dir.display()))?;

    // Write summary metadata
    let summary_path = base_dir.join("_summary.json");
    let summary_json = serde_json::to_string_pretty(&report.summary)?;
    std::fs::write(&summary_path, summary_json)?;

    let schema = create_repo_schema();

    // Process languages in parallel and write partitioned files
    report
        .languages
        .par_iter()
        .try_for_each(|(language, lang_report)| -> Result<()> {
            let lang_dir = base_dir.join("language").join(language);
            create_dir_all(&lang_dir).with_context(|| {
                format!("Failed to create language directory: {}", lang_dir.display())
            })?;

            // Convert language data to RepoRecord format
            let mut records: Vec<RepoRecord> = lang_report
                .repo_stats
                .iter()
                .map(|(repo_name, stats)| RepoRecord::from_repo_stats(language, repo_name, stats))
                .collect();

            // Sort by average_score for better compression and query performance
            records.sort_by(|a, b| {
                a.average_score
                    .partial_cmp(&b.average_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Partition by score ranges for parallel processing
            let chunk_size = (records.len() / rayon::current_num_threads()).max(1000);
            let chunks: Vec<_> = records.chunks(chunk_size).enumerate().collect();

            chunks
                .par_iter()
                .try_for_each(|(chunk_idx, chunk)| -> Result<()> {
                    let partition_file = lang_dir.join(format!("part_{:04}.parquet", chunk_idx));
                    write_records_to_parquet(chunk, &schema, &partition_file)?;
                    Ok(())
                })?;

            println!(
                "  ✓ Written {} - {} repos in {} partitions",
                language,
                records.len(),
                chunks.len()
            );

            Ok(())
        })?;

    println!("Arrow/Parquet report written to: {}", base_dir.display());
    Ok(())
}

fn process_language_directory(dir_path: &Path) -> Result<LanguageReport> {
    // Collect all *.jsonl.zst files first
    let files: Vec<PathBuf> = WalkDir::new(dir_path)
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            if path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("zst")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with(".jsonl"))
                    .unwrap_or(false)
            {
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
        println!(
            "  [{}/{}] Processing file: {}",
            current,
            files.len(),
            file_path.display()
        );

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
                        repo_scores_map
                            .entry(repo_name)
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
            let min_score = scores
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_score = scores
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let average_score = total_score / scores.len() as f64;

            repo_stats.insert(
                repo_name,
                RepoStats {
                    document_count: scores.len(),
                    total_score,
                    average_score,
                    min_score,
                    max_score,
                },
            );
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
            format!(
                "Failed to read line {} from: {}",
                line_num + 1,
                file_path.display()
            )
        })?;

        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<Document>(&line) {
            Ok(doc) => {
                document_count += 1;

                if let (Some(repo_name), Some(score)) = (doc.metadata.repo_name, doc.metadata.score) {
                    repo_scores
                        .entry(repo_name)
                        .or_insert_with(Vec::new)
                        .push(score);
                }
            }
            Err(e) => {
                eprintln!(
                    "    Warning: Failed to parse JSON on line {}: {}",
                    line_num + 1,
                    e
                );
            }
        }
    }

    Ok((document_count, repo_scores))
}