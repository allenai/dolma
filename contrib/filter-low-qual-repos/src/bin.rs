use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::types::*;
use crate::utils::*;

pub fn bin_repositories(
    source_dir: Option<PathBuf>,
    report_path: PathBuf,
    build_report: bool,
    num_bins: usize,
    sample_size: usize,
    output: PathBuf,
) -> Result<()> {
    let start_time = Instant::now();
    println!("Starting repository binning process...");

    // Handle report building or loading
    if build_report {
        let source_dir = source_dir
            .ok_or_else(|| anyhow::anyhow!("--source-dir is required when using --build-report"))?;

        if !source_dir.exists() {
            anyhow::bail!("Source directory does not exist: {}", source_dir.display());
        }

        println!("Building new score report from source directory...");
        build_score_report(source_dir, report_path.clone())?;
    }

    // Check if this is an Arrow-format report or legacy JSON
    let is_arrow_format = check_if_arrow_format(&report_path);

    if !is_arrow_format && !report_path.exists() {
        anyhow::bail!(
            "Score report file does not exist: {}. Use --build-report flag to create it from source data.",
            report_path.display()
        );
    }

    println!(
        "Using {} score report: {}",
        if is_arrow_format {
            "Arrow/Parquet"
        } else {
            "JSON"
        },
        report_path.display()
    );

    // Load summary based on format
    let summary = if is_arrow_format {
        load_arrow_summary(&report_path)?
    } else {
        load_score_report_summary(&report_path)?
    };

    // Create bins per language using appropriate approach
    println!(
        "Creating language-specific bins with {} processing...",
        if is_arrow_format {
            "parallel Arrow"
        } else {
            "streaming JSON"
        }
    );

    // Get available languages based on format
    let available_languages = if is_arrow_format {
        get_available_arrow_languages(&report_path)?
    } else {
        get_available_languages(&report_path)?
    };
    let total_languages = available_languages.len();
    println!("Processing {} languages in parallel", total_languages);

    // Process languages in parallel using appropriate method
    let language_bins: HashMap<String, LanguageBinReport> = available_languages
        .par_iter()
        .filter_map(|language| {
            println!("Processing language: {}", language);

            let result = if is_arrow_format {
                process_language_arrow(&report_path, language, num_bins, sample_size)
            } else {
                process_language_streaming(&report_path, language, num_bins, sample_size)
            };

            match result {
                Ok(Some(lang_bin_report)) => {
                    println!(
                        "  ✓ Completed {} - {} repos, {} bins",
                        language,
                        lang_bin_report.summary.total_repositories,
                        lang_bin_report.bins.len()
                    );
                    Some((language.clone(), lang_bin_report))
                }
                Ok(None) => {
                    println!("  No repositories found for {}, skipping", language);
                    None
                }
                Err(e) => {
                    eprintln!("  ✗ Error processing {}: {}", language, e);
                    None
                }
            }
        })
        .collect();

    // Calculate totals
    let total_repos: usize = language_bins
        .values()
        .map(|lb| lb.summary.total_repositories)
        .sum();
    let total_docs: usize = language_bins
        .values()
        .map(|lb| lb.summary.total_documents)
        .sum();

    let bin_report = BinReport {
        language_bins,
        summary: BinSummary {
            total_languages: summary.total_languages,
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
    println!(
        "  Sample size per bin: {}",
        bin_report.summary.sample_size_per_bin
    );
    println!("  Completed in: {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

fn build_score_report(source_dir: PathBuf, output_path: PathBuf) -> Result<()> {
    crate::aggregate::aggregate_scores(source_dir, output_path)
}

fn process_language_arrow(
    report_path: &PathBuf,
    language: &str,
    num_bins: usize,
    sample_size: usize,
) -> Result<Option<LanguageBinReport>> {
    let records = load_language_records_parallel(report_path, language)?;

    if records.is_empty() {
        return Ok(None);
    }

    println!("  {} repositories loaded from Arrow format", records.len());

    // Convert RepoRecord to the tuple format expected by create_language_bins_optimized
    let language_repos: Vec<(String, f64, usize)> = records
        .iter()
        .map(|r| {
            (
                r.repo_name.clone(),
                r.average_score,
                r.document_count as usize,
            )
        })
        .collect();

    let min_score = records.first().unwrap().average_score;
    let max_score = records.last().unwrap().average_score;

    println!(
        "  {} repositories, score range: {:.6} to {:.6}",
        language_repos.len(),
        min_score,
        max_score
    );

    // Create bins for this language
    let lang_bins = create_language_bins_optimized(
        &language_repos,
        num_bins,
        sample_size,
        min_score,
        max_score,
        language,
    )?;

    // Calculate total documents from records
    let total_documents: u64 = records.iter().map(|r| r.document_count).sum();

    let lang_bin_report = LanguageBinReport {
        language: language.to_string(),
        bins: lang_bins,
        summary: LanguageBinSummary {
            language: language.to_string(),
            total_repositories: language_repos.len(),
            total_documents: total_documents as usize,
            num_bins,
            sample_size_per_bin: sample_size,
        },
    };

    Ok(Some(lang_bin_report))
}

fn process_language_streaming(
    report_path: &Path,
    language: &str,
    num_bins: usize,
    sample_size: usize,
) -> Result<Option<LanguageBinReport>> {
    let file = File::open(report_path)
        .with_context(|| format!("Failed to open score report: {}", report_path.display()))?;

    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut found_language = false;
    let mut language_json = String::new();
    let mut brace_count = 0;
    let mut in_language_section = false;

    // Find the target language
    while reader.read_line(&mut line)? > 0 {
        if line.contains(&format!("\"{}\": {{", language)) {
            found_language = true;
            in_language_section = true;
            language_json = line.clone();
            brace_count = 1;
        } else if in_language_section {
            language_json.push_str(&line);
            for ch in line.chars() {
                if ch == '{' {
                    brace_count += 1;
                } else if ch == '}' {
                    brace_count -= 1;
                }
            }
        }

        if in_language_section && brace_count == 0 {
            break;
        }

        line.clear();
    }

    if !found_language {
        return Ok(None);
    }

    // Extract the JSON value for this language
    let start_pos = language_json.find('{').unwrap();
    let json_str = &language_json[start_pos..language_json.rfind('}').unwrap() + 1];

    let language_report: LanguageReport = serde_json::from_str(json_str)
        .with_context(|| format!("Failed to parse language report for {}", language))?;

    if language_report.repo_stats.is_empty() {
        return Ok(None);
    }

    // Extract repositories for this language
    let mut language_repos = Vec::with_capacity(language_report.repo_stats.len());
    for (repo_name, repo_stats) in &language_report.repo_stats {
        language_repos.push((
            repo_name.clone(),
            repo_stats.average_score,
            repo_stats.document_count,
        ));
    }

    // Sort repositories by average score
    language_repos
        .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let min_score = language_repos[0].1;
    let max_score = language_repos[language_repos.len() - 1].1;

    println!(
        "  {} repositories, score range: {:.6} to {:.6}",
        language_repos.len(),
        min_score,
        max_score
    );

    // Create bins for this language
    let lang_bins = create_language_bins_optimized(
        &language_repos,
        num_bins,
        sample_size,
        min_score,
        max_score,
        language,
    )?;

    let lang_bin_report = LanguageBinReport {
        language: language.to_string(),
        bins: lang_bins,
        summary: LanguageBinSummary {
            language: language.to_string(),
            total_repositories: language_repos.len(),
            total_documents: language_report.total_documents,
            num_bins,
            sample_size_per_bin: sample_size,
        },
    };

    Ok(Some(lang_bin_report))
}

fn create_language_bins_optimized(
    language_repos: &[(String, f64, usize)], // (repo_name, avg_score, doc_count) - assumed sorted
    num_bins: usize,
    sample_size: usize,
    min_score: f64,
    max_score: f64,
    language: &str, // Add language parameter
) -> Result<Vec<ScoreBin>> {
    if language_repos.is_empty() {
        return Ok(Vec::new());
    }

    let bin_width = (max_score - min_score) / num_bins as f64;

    // Pre-allocate bins vector
    let mut bins = Vec::with_capacity(num_bins);

    // Since repos are sorted by score, we can partition them efficiently
    let mut repo_index = 0;

    for i in 0..num_bins {
        let bin_min = min_score + i as f64 * bin_width;
        let bin_max = if i == num_bins - 1 {
            max_score
        } else {
            min_score + (i + 1) as f64 * bin_width
        };

        // Find start of bin using binary search (repos are sorted)
        let start_idx = if repo_index < language_repos.len() {
            repo_index
                + language_repos[repo_index..]
                    .binary_search_by(|(_, score, _)| {
                        if *score < bin_min {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                    .unwrap_or_else(|idx| idx)
        } else {
            language_repos.len()
        };

        // Find end of bin
        let end_idx = language_repos[start_idx..]
            .binary_search_by(|(_, score, _)| {
                if *score <= bin_max {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|idx| idx)
            + start_idx;

        let repos_in_bin = &language_repos[start_idx..end_idx];
        let total_repos_in_range = repos_in_bin.len();

        // Apply reservoir sampling with optimized allocation
        let sample_repos: Vec<BinRepo> = if repos_in_bin.len() <= sample_size {
            // Take all repos if fewer than sample size
            repos_in_bin
                .iter()
                .map(|(repo_name, avg_score, doc_count)| {
                    BinRepo {
                        repo_name: repo_name.clone(),
                        language: language.to_string(),
                        average_score: *avg_score,
                        document_count: *doc_count,
                    }
                })
                .collect()
        } else {
            // Reservoir sampling with pre-allocated vector
            let mut rng = thread_rng();
            let sampled: Vec<_> = repos_in_bin
                .choose_multiple(&mut rng, sample_size)
                .collect();
            let mut sample_repos = Vec::with_capacity(sample_size);
            for (repo_name, avg_score, doc_count) in sampled {
                sample_repos.push(BinRepo {
                    repo_name: repo_name.clone(),
                    language: language.to_string(),
                    average_score: *avg_score,
                    document_count: *doc_count,
                });
            }
            sample_repos
        };

        println!(
            "    Bin {}: [{:.6}, {:.6}] - {} repos total, {} sampled",
            i + 1,
            bin_min,
            bin_max,
            total_repos_in_range,
            sample_repos.len()
        );

        bins.push(ScoreBin {
            min_score: bin_min,
            max_score: bin_max,
            sample_repos,
            total_repos_in_range,
        });

        // Update repo_index for next iteration
        repo_index = end_idx;
        if repo_index >= language_repos.len() {
            // No more repos, fill remaining bins with empty bins
            for j in (i + 1)..num_bins {
                let empty_bin_min = min_score + j as f64 * bin_width;
                let empty_bin_max = if j == num_bins - 1 {
                    max_score
                } else {
                    min_score + (j + 1) as f64 * bin_width
                };

                bins.push(ScoreBin {
                    min_score: empty_bin_min,
                    max_score: empty_bin_max,
                    sample_repos: Vec::new(),
                    total_repos_in_range: 0,
                });

                println!(
                    "    Bin {}: [{:.6}, {:.6}] - 0 repos total, 0 sampled",
                    j + 1,
                    empty_bin_min,
                    empty_bin_max
                );
            }
            break;
        }
    }

    Ok(bins)
}