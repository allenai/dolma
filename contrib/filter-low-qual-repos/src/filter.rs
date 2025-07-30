use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicUsize, atomic::Ordering};
use std::time::Instant;
use walkdir::WalkDir;

use crate::types::*;
use crate::utils::*;

pub fn filter_documents(
    source_dir: PathBuf,
    report_path: PathBuf,
    bin_path: PathBuf,
    target_bins: Vec<usize>,
    target_language: Option<String>,
    output_dir: PathBuf,
    _max_file_size_mb: usize,
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
    std::fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    // Load bin report to get target repositories
    println!("Loading bin report...");
    let bin_report = load_bin_report(&bin_path)?;

    // Extract target repositories using appropriate method based on format
    let is_arrow_format = check_if_arrow_format(&report_path);
    println!(
        "Using {} format for repository extraction",
        if is_arrow_format {
            "Arrow/Parquet"
        } else {
            "JSON streaming"
        }
    );

    let target_repos = if is_arrow_format {
        extract_target_repos_arrow(&bin_report, &report_path, &target_bins, &target_language)?
    } else {
        extract_target_repos_streaming_by_language(
            &bin_report,
            &report_path,
            &target_bins,
            &target_language,
        )?
    };
    println!(
        "Found {} target repositories across {} bins",
        target_repos.len(),
        target_bins.len()
    );

    // Get all source files with their language directories
    println!("Scanning source files...");
    let source_files = collect_source_files_with_language(&source_dir)?;
    println!(
        "Found {} source files across {} languages to process",
        source_files.len(),
        source_files
            .iter()
            .map(|(lang, _)| lang)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    // Setup progress bar
    let pb = ProgressBar::new(source_files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({percent}%) {msg}")
        .unwrap()
        .progress_chars("#>-"));

    // Process files individually to maintain 1:1 or 1:0 input:output relationship
    let processed_docs = AtomicUsize::new(0);
    let filtered_docs = AtomicUsize::new(0);
    let output_files_created = AtomicUsize::new(0);

    // Process each file individually in parallel
    source_files.par_iter().for_each(|(language, file_path)| {
        match process_and_write_source_file(
            file_path,
            language,
            &target_repos,
            &output_dir,
            &processed_docs,
            &filtered_docs,
            &output_files_created,
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error processing {}: {}", file_path.display(), e);
            }
        }
        pb.inc(1);
    });

    pb.finish_with_message("Processing complete");

    let total_output_files = output_files_created.load(Ordering::Relaxed);

    let elapsed = start_time.elapsed();
    let total_processed = processed_docs.load(Ordering::Relaxed);
    let total_retained = filtered_docs.load(Ordering::Relaxed);
    let total_excluded = total_processed - total_retained;

    println!("Filter completed in {:.2}s", elapsed.as_secs_f64());
    println!("Summary:");
    println!("  Documents processed: {}", total_processed);
    println!("  Documents retained: {}", total_retained);
    println!("  Documents excluded: {}", total_excluded);
    println!(
        "  Retention rate: {:.2}%",
        if total_processed > 0 {
            (total_retained as f64 / total_processed as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("  Output files: {}", total_output_files);
    println!("  Output directory: {}", output_dir.display());

    Ok(())
}

fn extract_target_repos_arrow(
    bin_report: &BinReport,
    report_path: &PathBuf,
    target_bins: &[usize],
    target_language: &Option<String>,
) -> Result<HashSet<String>> {
    println!("Extracting target repositories from Arrow/Parquet report...");

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
                    anyhow::bail!(
                        "Invalid bin number: {} for language {}. Valid range: 1-{}",
                        bin_idx,
                        language,
                        lang_bin_report.bins.len()
                    );
                }

                let bin = &lang_bin_report.bins[bin_idx - 1]; // Convert to 0-indexed
                score_ranges.push((bin.min_score, bin.max_score));
                println!(
                    "Target bin {} for {}: score range [{:.6}, {:.6}]",
                    bin_idx, language, bin.min_score, bin.max_score
                );
            }

            language_score_ranges.insert(language.clone(), score_ranges);
        } else {
            if target_language.is_some() {
                anyhow::bail!("Language '{}' not found in bin report", language);
            }
            println!(
                "Warning: Language '{}' not found in bin report, skipping",
                language
            );
        }
    }

    // Load repositories from Arrow format and filter by score ranges in parallel
    let lang_results: Result<Vec<HashSet<String>>> = language_score_ranges
        .par_iter()
        .map(|(language, score_ranges)| -> Result<HashSet<String>> {
            println!(
                "  Processing language: {} with {} score ranges",
                language,
                score_ranges.len()
            );

            let records = load_language_records_parallel(report_path, language)?;
            let mut lang_target_repos = HashSet::new();

            for record in records {
                // Check if repository falls within any target score range for this language
                for &(min_score, max_score) in score_ranges {
                    if record.average_score >= min_score && record.average_score <= max_score {
                        lang_target_repos.insert(record.repo_name.clone());
                        break;
                    }
                }
            }

            println!(
                "  Completed {} - found {} matching repos",
                language,
                lang_target_repos.len()
            );
            Ok(lang_target_repos)
        })
        .collect();

    // Combine all language results
    let mut target_repos = HashSet::new();
    for lang_repos in lang_results? {
        target_repos.extend(lang_repos);
    }

    println!(
        "  Arrow extraction completed - found {} unique repositories total",
        target_repos.len()
    );
    Ok(target_repos)
}

fn extract_target_repos_streaming_by_language(
    bin_report: &BinReport,
    report_path: &Path,
    target_bins: &[usize],
    target_language: &Option<String>,
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
                    anyhow::bail!(
                        "Invalid bin number: {} for language {}. Valid range: 1-{}",
                        bin_idx,
                        language,
                        lang_bin_report.bins.len()
                    );
                }

                let bin = &lang_bin_report.bins[bin_idx - 1]; // Convert to 0-indexed
                score_ranges.push((bin.min_score, bin.max_score));
                println!(
                    "Target bin {} for {}: score range [{:.6}, {:.6}]",
                    bin_idx, language, bin.min_score, bin.max_score
                );
            }

            language_score_ranges.insert(language.clone(), score_ranges);
        } else {
            if target_language.is_some() {
                anyhow::bail!("Language '{}' not found in bin report", language);
            }
            println!(
                "Warning: Language '{}' not found in bin report, skipping",
                language
            );
        }
    }

    let file = File::open(report_path)
        .with_context(|| format!("Failed to open score report: {}", report_path.display()))?;

    parse_score_report_for_language_ranges(file, &language_score_ranges)
}

fn collect_source_files_with_language(source_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut source_files = Vec::new();

    // Check if source_dir contains *.jsonl.zst files directly (single language mode)
    let has_jsonl_files = std::fs::read_dir(source_dir)?
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

        for entry in WalkDir::new(source_dir) {
            let entry = entry?;
            let path = entry.path();

            if path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("zst")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with(".jsonl"))
                    .unwrap_or(false)
            {
                source_files.push((language_name.clone(), path.to_path_buf()));
            }
        }
    } else {
        // Multi-language directory mode
        for entry in std::fs::read_dir(source_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let language_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Collect files from this language directory
                for file_entry in WalkDir::new(&path) {
                    let file_entry = file_entry?;
                    let file_path = file_entry.path();

                    if file_path.is_file()
                        && file_path.extension().and_then(|s| s.to_str()) == Some("zst")
                        && file_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.ends_with(".jsonl"))
                            .unwrap_or(false)
                    {
                        source_files.push((language_name.clone(), file_path.to_path_buf()));
                    }
                }
            }
        }
    }

    Ok(source_files)
}

fn process_and_write_source_file(
    file_path: &Path,
    language: &str,
    target_repos: &HashSet<String>,
    output_dir: &Path,
    processed_docs: &AtomicUsize,
    filtered_docs: &AtomicUsize,
    output_files_created: &AtomicUsize,
) -> Result<()> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;

    let decoder = zstd::Decoder::new(file)
        .with_context(|| format!("Failed to create zstd decoder for: {}", file_path.display()))?;

    let reader = BufReader::new(decoder);
    let mut filtered_lines = Vec::new();

    // Filter documents from this file
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
                    filtered_lines.push(line);
                    filtered_docs.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    // Only create output file if we have filtered content
    if !filtered_lines.is_empty() {
        let language_output_dir = output_dir.join(language);
        std::fs::create_dir_all(&language_output_dir)
            .with_context(|| format!("Failed to create language directory: {}", language_output_dir.display()))?;

        // Generate output filename based on input filename
        let input_filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.jsonl.zst");
        
        let output_file_path = language_output_dir.join(input_filename);
        
        let output_file = File::create(&output_file_path)
            .with_context(|| format!("Failed to create output file: {}", output_file_path.display()))?;

        let mut encoder = zstd::Encoder::new(output_file, 3)?;
        
        for line in filtered_lines {
            encoder.write_all(line.as_bytes())?;
            encoder.write_all(b"\n")?;
        }
        
        encoder.finish()?;
        output_files_created.fetch_add(1, Ordering::Relaxed);
    }

    Ok(())
}

