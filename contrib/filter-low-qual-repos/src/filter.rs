use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicUsize, atomic::Ordering, mpsc};
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

    // Process files and collect matching documents with parallel output writing
    let max_file_size_bytes = max_file_size_mb * 1024 * 1024;
    let processed_docs = AtomicUsize::new(0);
    let filtered_docs = AtomicUsize::new(0);

    // Create channel for batched output communication
    let (tx, rx) = mpsc::channel::<(String, HashMap<String, Vec<String>>)>();
    
    // Spawn dedicated output writer thread
    let output_dir_clone = output_dir.clone();
    let writer_handle = std::thread::spawn(move || -> Result<OutputManager> {
        let mut output_manager = OutputManager::new(output_dir_clone, max_file_size_bytes)?;
        
        while let Ok((language, repo_docs)) = rx.recv() {
            if let Err(e) = output_manager.write_repo_documents_for_language(&language, repo_docs) {
                eprintln!("Error writing output for language {}: {}", language, e);
            }
        }
        
        output_manager.finalize()?;
        Ok(output_manager)
    });

    source_files.par_iter().for_each(|(language, file_path)| {
        match process_source_file_with_language_buffered(
            file_path,
            language,
            &target_repos,
            &tx,
            &processed_docs,
            &filtered_docs,
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error processing {}: {}", file_path.display(), e);
            }
        }
        pb.inc(1);
    });

    pb.finish_with_message("Processing complete");

    // Close the channel and wait for output writer to finish
    drop(tx);
    let output_mgr = writer_handle.join().expect("Output writer thread panicked")?;

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
    println!("  Output files: {}", output_mgr.file_count());
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

fn process_source_file_with_language_buffered(
    file_path: &Path,
    language: &str,
    target_repos: &HashSet<String>,
    tx: &mpsc::Sender<(String, HashMap<String, Vec<String>>)>,
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
                    repo_docs
                        .entry(repo_name)
                        .or_insert_with(Vec::new)
                        .push(line);
                    filtered_docs.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    // Send documents to output writer thread if we have any
    if !repo_docs.is_empty() {
        if let Err(e) = tx.send((language.to_string(), repo_docs)) {
            eprintln!("Failed to send output data for {}: {}", language, e);
        }
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

    fn write_repo_documents_for_language(
        &mut self,
        language: &str,
        repo_docs: HashMap<String, Vec<String>>,
    ) -> Result<()> {
        // Get or create language manager
        if !self.language_managers.contains_key(language) {
            let language_dir = self.output_dir.join(language);
            std::fs::create_dir_all(&language_dir).with_context(|| {
                format!(
                    "Failed to create language directory: {}",
                    language_dir.display()
                )
            })?;

            self.language_managers.insert(
                language.to_string(),
                LanguageOutputManager {
                    language_dir,
                    current_file_idx: 0,
                    current_writer: None,
                    current_size: 0,
                    file_count: 0,
                },
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
    fn write_repo_documents(
        &mut self,
        repo_docs: HashMap<String, Vec<String>>,
        max_file_size: usize,
    ) -> Result<()> {
        // Sort repositories for consistent output
        let mut sorted_repos: Vec<_> = repo_docs.into_iter().collect();
        sorted_repos.sort_by(|a, b| a.0.cmp(&b.0));

        for (_repo_name, docs) in sorted_repos {
            // Calculate size needed for this repository
            let repo_size: usize = docs.iter().map(|doc| doc.len() + 1).sum(); // +1 for newline

            // Check if we need a new file
            if self.current_writer.is_none()
                || (self.current_size + repo_size > max_file_size && self.current_size > 0)
            {
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
        let file_path = self
            .language_dir
            .join(format!("{:06}.jsonl.zst", self.current_file_idx));
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