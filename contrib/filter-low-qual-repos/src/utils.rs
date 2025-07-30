use anyhow::{Context, Result};
use arrow::array::{Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::types::*;

pub fn create_repo_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("language", DataType::Utf8, false),
        Field::new("repo_name", DataType::Utf8, false),
        Field::new("document_count", DataType::UInt64, false),
        Field::new("total_score", DataType::Float64, false),
        Field::new("average_score", DataType::Float64, false),
        Field::new("min_score", DataType::Float64, false),
        Field::new("max_score", DataType::Float64, false),
    ]))
}

pub fn write_records_to_parquet(
    records: &[RepoRecord],
    schema: &Arc<Schema>,
    file_path: &PathBuf,
) -> Result<()> {
    if records.is_empty() {
        return Ok(());
    }

    // Create Arrow arrays from records
    let language_array = StringArray::from_iter_values(records.iter().map(|r| &r.language));
    let repo_name_array = StringArray::from_iter_values(records.iter().map(|r| &r.repo_name));
    let document_count_array = UInt64Array::from_iter_values(records.iter().map(|r| r.document_count));
    let total_score_array = Float64Array::from_iter_values(records.iter().map(|r| r.total_score));
    let average_score_array = Float64Array::from_iter_values(records.iter().map(|r| r.average_score));
    let min_score_array = Float64Array::from_iter_values(records.iter().map(|r| r.min_score));
    let max_score_array = Float64Array::from_iter_values(records.iter().map(|r| r.max_score));

    // Create record batch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(language_array),
            Arc::new(repo_name_array),
            Arc::new(document_count_array),
            Arc::new(total_score_array),
            Arc::new(average_score_array),
            Arc::new(min_score_array),
            Arc::new(max_score_array),
        ],
    )?;

    // Write to Parquet file
    let file = File::create(file_path)
        .with_context(|| format!("Failed to create parquet file: {}", file_path.display()))?;

    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

pub fn load_arrow_summary(report_path: &PathBuf) -> Result<Summary> {
    let base_dir = if report_path.extension().is_some() {
        report_path.with_extension("")
    } else {
        report_path.clone()
    };

    let summary_path = base_dir.join("_summary.json");
    let summary_json = std::fs::read_to_string(&summary_path)
        .with_context(|| format!("Failed to read summary file: {}", summary_path.display()))?;

    let summary: Summary = serde_json::from_str(&summary_json)
        .with_context(|| "Failed to parse summary JSON")?;

    Ok(summary)
}

pub fn get_available_arrow_languages(report_path: &PathBuf) -> Result<Vec<String>> {
    let base_dir = if report_path.extension().is_some() {
        report_path.with_extension("")
    } else {
        report_path.clone()
    };

    let language_dir = base_dir.join("language");
    if !language_dir.exists() {
        return Ok(Vec::new());
    }

    let mut languages = Vec::new();
    for entry in std::fs::read_dir(&language_dir)? {
        let entry = entry?;
        if entry.path().is_dir() {
            if let Some(lang_name) = entry.file_name().to_str() {
                languages.push(lang_name.to_string());
            }
        }
    }

    languages.sort();
    Ok(languages)
}

pub fn load_language_records_parallel(
    report_path: &PathBuf,
    language: &str,
) -> Result<Vec<RepoRecord>> {
    let base_dir = if report_path.extension().is_some() {
        report_path.with_extension("")
    } else {
        report_path.clone()
    };

    let lang_dir = base_dir.join("language").join(language);
    if !lang_dir.exists() {
        return Ok(Vec::new());
    }

    // Collect all parquet files for this language
    let mut parquet_files = Vec::new();
    for entry in std::fs::read_dir(&lang_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
            parquet_files.push(path);
        }
    }

    if parquet_files.is_empty() {
        return Ok(Vec::new());
    }

    println!(
        "  Loading {} partition files for {} in parallel",
        parquet_files.len(),
        language
    );

    // Load partitions in parallel
    use rayon::prelude::*;
    let all_records: Vec<Vec<RepoRecord>> = parquet_files
        .par_iter()
        .filter_map(|file_path| match load_parquet_records(file_path) {
            Ok(records) => Some(records),
            Err(e) => {
                eprintln!("  Warning: Failed to load {}: {}", file_path.display(), e);
                None
            }
        })
        .collect();

    // Flatten all records
    let mut records: Vec<RepoRecord> = all_records.into_iter().flatten().collect();

    // Sort by average score for consistent ordering
    records.sort_by(|a, b| {
        a.average_score
            .partial_cmp(&b.average_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(records)
}

pub fn load_parquet_records(file_path: &PathBuf) -> Result<Vec<RepoRecord>> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open parquet file: {}", file_path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let record_batch_reader = builder.build()?;

    let mut records = Vec::new();

    for batch_result in record_batch_reader {
        let batch = batch_result?;

        // Extract arrays from the batch
        let language_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast language column"))?;
        let repo_name_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast repo_name column"))?;
        let document_count_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast document_count column"))?;
        let total_score_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast total_score column"))?;
        let average_score_array = batch
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast average_score column"))?;
        let min_score_array = batch
            .column(5)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast min_score column"))?;
        let max_score_array = batch
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast max_score column"))?;

        // Convert to RepoRecord structs
        for i in 0..batch.num_rows() {
            let record = RepoRecord {
                language: language_array.value(i).to_string(),
                repo_name: repo_name_array.value(i).to_string(),
                document_count: document_count_array.value(i),
                total_score: total_score_array.value(i),
                average_score: average_score_array.value(i),
                min_score: min_score_array.value(i),
                max_score: max_score_array.value(i),
            };
            records.push(record);
        }
    }

    Ok(records)
}

pub fn check_if_arrow_format(report_path: &PathBuf) -> bool {
    let base_dir = if report_path.extension().is_some() {
        report_path.with_extension("")
    } else {
        report_path.clone()
    };

    // Check if Arrow format directory structure exists
    base_dir.join("_summary.json").exists() && base_dir.join("language").exists()
}

pub fn load_score_report_summary(report_path: &Path) -> Result<Summary> {
    let file = File::open(report_path)
        .with_context(|| format!("Failed to open score report: {}", report_path.display()))?;

    let mut reader = BufReader::new(file);
    let mut line = String::new();

    // Skip to summary section
    while reader.read_line(&mut line)? > 0 {
        if line.trim().starts_with("\"summary\":") {
            // Parse just the summary
            let summary_start = line.find('{').unwrap();
            let mut summary_json = line[summary_start..].to_string();

            // Read until we find the closing brace
            let mut brace_count = 1;
            line.clear();
            while brace_count > 0 && reader.read_line(&mut line)? > 0 {
                for ch in line.chars() {
                    if ch == '{' {
                        brace_count += 1;
                    } else if ch == '}' {
                        brace_count -= 1;
                    }
                }
                summary_json.push_str(&line);
                line.clear();
            }

            // Extract just the summary object
            let end_pos = summary_json.rfind('}').unwrap() + 1;
            let summary_json = &summary_json[..end_pos];

            let summary: Summary = serde_json::from_str(summary_json)
                .with_context(|| "Failed to parse summary from score report")?;

            return Ok(summary);
        }
        line.clear();
    }

    anyhow::bail!("Could not find summary in score report")
}

pub fn get_available_languages(report_path: &Path) -> Result<Vec<String>> {
    let file = File::open(report_path)
        .with_context(|| format!("Failed to open score report: {}", report_path.display()))?;

    let mut reader = BufReader::new(file);
    let mut languages = Vec::new();
    let mut line = String::new();
    let mut in_languages_section = false;

    while reader.read_line(&mut line)? > 0 {
        if line.trim().starts_with("\"languages\":") {
            in_languages_section = true;
        } else if in_languages_section && line.trim().starts_with('"') && line.contains(":") {
            // Extract language name from line like "  "language_name": {"
            let start = line.find('"').unwrap() + 1;
            let end = line[start..].find('"').unwrap();
            let language = line[start..start + end].to_string();
            languages.push(language);
        } else if in_languages_section && line.trim() == "}" {
            break;
        }
        line.clear();
    }

    Ok(languages)
}

pub fn load_bin_report(bin_path: &Path) -> Result<BinReport> {
    let file = File::open(bin_path)
        .with_context(|| format!("Failed to open bin report: {}", bin_path.display()))?;

    let report: BinReport = serde_json::from_reader(file)
        .with_context(|| format!("Failed to parse bin report: {}", bin_path.display()))?;

    Ok(report)
}

pub fn parse_score_report_for_language_ranges(
    file: File,
    language_score_ranges: &HashMap<String, Vec<(f64, f64)>>,
) -> Result<HashSet<String>> {
    let mut target_repos = HashSet::new();
    let reader = BufReader::new(file);

    // Use serde_json's streaming API to parse the structure
    let mut deserializer = serde_json::Deserializer::from_reader(reader);

    // Parse the JSON structure piece by piece
    let value: serde_json::Value = Deserialize::deserialize(&mut deserializer)?;

    if let Some(languages) = value.get("languages").and_then(|v| v.as_object()) {
        let mut processed_repos = 0;
        let total_languages = languages.len();

        for (lang_idx, (language_name, language_data)) in languages.iter().enumerate() {
            if lang_idx % 10 == 0 {
                println!(
                    "  Processing language {}/{} - found {} repos so far",
                    lang_idx + 1,
                    total_languages,
                    target_repos.len()
                );
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
                            println!(
                                "  Processed {} repositories, found {} matches",
                                processed_repos,
                                target_repos.len()
                            );
                        }
                    }
                }
            }
        }

        println!("  Completed processing {} repositories total", processed_repos);
    }

    Ok(target_repos)
}