use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "to-dolma")]
#[command(about = "Convert JSONL.zst files from one schema to Dolma schema")]
struct Args {
    /// Input directory containing JSONL.zst files
    #[arg(short, long)]
    input_dir: PathBuf,
    
    /// Output directory for converted files
    #[arg(short, long)]
    output_dir: PathBuf,
    
    /// Number of parallel workers (default: number of CPU cores)
    #[arg(short, long)]
    workers: Option<usize>,
}

#[derive(Deserialize)]
struct InputRecord {
    blob_id: String,
    language: String,
    repo_name: String,
    path: String,
    src_encoding: String,
    length_bytes: u64,
    score: f64,
    int_score: u32,
    detected_licenses: Vec<String>,
    license_type: String,
    text: String,
    download_success: bool,
}

#[derive(Serialize)]
struct OutputMetadata {
    uri: String,
    length_bytes: u64,
    score: f64,
    src_encoding: String,
    repo_name: String,
    detected_licenses: Vec<String>,
    int_score: u32,
    path: String,
    language: String,
    license_type: String,
}

#[derive(Serialize)]
struct OutputRecord {
    id: String,
    text: String,
    source: String,
    created: String,
    added: String,
    metadata: OutputMetadata,
}

impl From<InputRecord> for OutputRecord {
    fn from(input: InputRecord) -> Self {
        let metadata = OutputMetadata {
            uri: format!(
                "s3://softwareheritage/content/{{'blob_id': '{}', 'language': '{}', 'repo_name': '{}', 'path': '{}', 'src_encoding': '{}', 'length_bytes': {}, 'score': {}, 'int_score': {}, 'detected_licenses': {:?}, 'license_type': '{}'}}",
                input.blob_id,
                input.language,
                input.repo_name,
                input.path,
                input.src_encoding,
                input.length_bytes,
                input.score,
                input.int_score,
                input.detected_licenses,
                input.license_type
            ),
            length_bytes: input.length_bytes,
            score: input.score,
            src_encoding: input.src_encoding,
            repo_name: input.repo_name,
            detected_licenses: input.detected_licenses,
            int_score: input.int_score,
            path: input.path,
            language: input.language,
            license_type: input.license_type,
        };

        OutputRecord {
            id: input.blob_id,
            text: input.text,
            source: String::new(),
            created: String::new(),
            added: String::new(),
            metadata,
        }
    }
}

fn process_file(input_path: &Path, output_path: &Path) -> Result<(usize, usize)> {
    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Open input file with zstd decompression
    let input_file = std::fs::File::open(input_path)?;
    let decoder = zstd::Decoder::new(input_file)?;
    let reader = BufReader::new(decoder);

    // Open output file with zstd compression
    let output_file = std::fs::File::create(output_path)?;
    let encoder = zstd::Encoder::new(output_file, 3)?; // compression level 3
    let mut writer = BufWriter::new(encoder);

    let mut processed = 0;
    let mut errors = 0;

    for line in reader.lines() {
        let line = line?;
        
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<InputRecord>(&line) {
            Ok(input_record) => {
                let output_record: OutputRecord = input_record.into();
                let output_line = serde_json::to_string(&output_record)?;
                writeln!(writer, "{}", output_line)?;
                processed += 1;
            }
            Err(e) => {
                eprintln!("Error parsing line {} in {}: {}", processed + errors + 1, input_path.display(), e);
                errors += 1;
            }
        }
    }

    writer.flush()?;
    let encoder = writer.into_inner().map_err(|e| anyhow::anyhow!("Failed to get encoder: {}", e))?;
    encoder.finish()?;

    Ok((processed, errors))
}

fn find_jsonl_zst_files(input_dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            entry.path()
                .extension()
                .map_or(false, |ext| ext == "zst") &&
            entry.path()
                .file_stem()
                .and_then(|stem| Path::new(stem).extension())
                .map_or(false, |ext| ext == "jsonl")
        })
        .map(|entry| entry.path().to_path_buf())
        .collect()
}

fn relative_path(base: &Path, path: &Path) -> Result<PathBuf> {
    path.strip_prefix(base)
        .map(|p| p.to_path_buf())
        .map_err(|e| anyhow::anyhow!("Failed to get relative path: {}", e))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set up thread pool
    if let Some(workers) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()?;
    }

    // Find all .jsonl.zst files in input directory
    let input_files = find_jsonl_zst_files(&args.input_dir);
    
    if input_files.is_empty() {
        eprintln!("No .jsonl.zst files found in {}", args.input_dir.display());
        return Ok(());
    }

    eprintln!("Found {} .jsonl.zst files to process", input_files.len());

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Global counters
    let total_processed = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let files_completed = AtomicUsize::new(0);

    // Process files in parallel
    input_files.par_iter().try_for_each(|input_path| -> Result<()> {
        let rel_path = relative_path(&args.input_dir, input_path)?;
        let output_path = args.output_dir.join(&rel_path);

        eprintln!("Processing: {} -> {}", 
                 input_path.display(), 
                 output_path.display());

        let (processed, errors) = process_file(input_path, &output_path)?;
        
        total_processed.fetch_add(processed, Ordering::Relaxed);
        total_errors.fetch_add(errors, Ordering::Relaxed);
        let completed = files_completed.fetch_add(1, Ordering::Relaxed) + 1;
        
        eprintln!("Completed {} ({}/{}): {} records, {} errors", 
                 rel_path.display(),
                 completed,
                 input_files.len(),
                 processed, 
                 errors);

        Ok(())
    })?;

    let final_processed = total_processed.load(Ordering::Relaxed);
    let final_errors = total_errors.load(Ordering::Relaxed);

    eprintln!("Conversion complete!");
    eprintln!("Files processed: {}", input_files.len());
    eprintln!("Total records: {}", final_processed);
    eprintln!("Total errors: {}", final_errors);

    Ok(())
}