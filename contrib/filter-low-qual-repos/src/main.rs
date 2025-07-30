use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod types;
mod utils;
mod aggregate;
mod bin;
mod filter;
mod pipeline;

use aggregate::aggregate_scores;
use bin::bin_repositories;
use filter::filter_documents;
use pipeline::run_pipeline;

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
    /// Run the complete pipeline: aggregate -> bin -> filter
    Pipeline {
        /// Source directory containing programming language subdirectories with *.jsonl.zst files
        #[arg(short, long)]
        source_dir: PathBuf,

        /// Target bin numbers to include (comma-separated, 1-indexed)
        #[arg(short, long, value_delimiter = ',')]
        target_bins: Vec<usize>,

        /// Specific language to filter (optional, filters all languages if not specified)
        #[arg(short, long)]
        language: Option<String>,

        /// Output directory for filtered *.jsonl.zst files
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Number of bins to create
        #[arg(short, long, default_value = "10")]
        num_bins: usize,

        /// Sample size per bin using reservoir sampling
        #[arg(long, default_value = "100")]
        sample_size: usize,

        /// Maximum file size in MB before splitting
        #[arg(long, default_value = "50")]
        max_file_size_mb: usize,

        /// Working directory for intermediate reports
        #[arg(short, long, default_value = ".")]
        work_dir: PathBuf,
    },
}


fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Aggregate { source_dir, output } => aggregate_scores(source_dir, output),
        Commands::Bin {
            source_dir,
            report_path,
            build_report,
            num_bins,
            sample_size,
            output,
        } => bin_repositories(
            source_dir,
            report_path,
            build_report,
            num_bins,
            sample_size,
            output,
        ),
        Commands::Filter {
            source_dir,
            report_path,
            bin_path,
            target_bins,
            language,
            output_dir,
            max_file_size_mb,
        } => filter_documents(
            source_dir,
            report_path,
            bin_path,
            target_bins,
            language,
            output_dir,
            max_file_size_mb,
        ),
        Commands::Pipeline {
            source_dir,
            target_bins,
            language,
            output_dir,
            num_bins,
            sample_size,
            max_file_size_mb,
            work_dir,
        } => run_pipeline(
            source_dir,
            target_bins,
            language,
            output_dir,
            num_bins,
            sample_size,
            max_file_size_mb,
            work_dir,
        ),
    }
}


















