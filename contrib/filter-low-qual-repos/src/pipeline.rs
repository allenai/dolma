use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;

use crate::aggregate::aggregate_scores;
use crate::bin::bin_repositories;
use crate::filter::filter_documents;

pub fn run_pipeline(
    source_dir: PathBuf,
    target_bins: Vec<usize>,
    language: Option<String>,
    output_dir: PathBuf,
    num_bins: usize,
    sample_size: usize,
    max_file_size_mb: usize,
    work_dir: PathBuf,
) -> Result<()> {
    let start_time = Instant::now();
    println!("🚀 Starting complete pipeline: aggregate -> bin -> filter");
    println!("Source: {}", source_dir.display());
    println!("Output: {}", output_dir.display());
    println!("Work dir: {}", work_dir.display());

    // Create work directory if it doesn't exist
    std::fs::create_dir_all(&work_dir)
        .with_context(|| format!("Failed to create work directory: {}", work_dir.display()))?;

    // Define report paths with intelligent naming
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let score_report_path = work_dir.join(format!("repo_scores_{}.json", timestamp));
    let bin_report_path = work_dir.join(format!("repo_bins_{}.json", timestamp));

    println!("\n📊 Step 1/3: Aggregating repository scores...");
    println!("Report will be saved to: {}", score_report_path.display());

    // Step 1: Run aggregate
    aggregate_scores(source_dir.clone(), score_report_path.clone())?;

    println!("\n🗂️  Step 2/3: Creating score-based bins...");
    println!("Bin report will be saved to: {}", bin_report_path.display());

    // Step 2: Run bin
    bin_repositories(
        None, // Don't rebuild - use existing report
        score_report_path.clone(),
        false, // Don't force rebuild
        num_bins,
        sample_size,
        bin_report_path.clone(),
    )?;

    println!("\n🔍 Step 3/3: Filtering documents...");
    if let Some(ref lang) = language {
        println!("Filtering language: {}", lang);
    } else {
        println!("Filtering all languages");
    }
    println!("Target bins: {:?}", target_bins);

    // Step 3: Run filter
    filter_documents(
        source_dir,
        score_report_path.clone(),
        bin_report_path.clone(),
        target_bins,
        language,
        output_dir.clone(),
        max_file_size_mb,
    )?;

    let elapsed = start_time.elapsed();
    println!("\n✅ Pipeline completed in {:.2}s", elapsed.as_secs_f64());
    println!("📁 Intermediate reports:");
    println!("  Score report: {}", score_report_path.display());
    println!("  Bin report: {}", bin_report_path.display());
    println!("📁 Filtered output: {}", output_dir.display());

    // Optionally clean up intermediate reports
    println!("\n💡 Tip: You can reuse the reports for additional filtering:");
    println!("  cargo run -- filter \\");
    println!("    --report-path {} \\", score_report_path.display());
    println!("    --bin-path {} \\", bin_report_path.display());
    println!("    --target-bins <BINS> \\");
    println!("    --output-dir <OUTPUT>");

    Ok(())
}