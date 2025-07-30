use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{info, warn};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;
use zstd::stream::write::Encoder as ZstdEncoder;

#[derive(Parser)]
#[command(name = "make-ordered")]
#[command(about = "Groups documents by repository name and outputs ordered JSONL.zst files")]
struct Args {
    #[arg(help = "Input directory containing documents")]
    input_dir: PathBuf,
    
    #[arg(short, long, help = "Output directory for grouped files")]
    output_dir: PathBuf,
    
    #[arg(
        short, 
        long, 
        default_value = "104857600", 
        help = "Target file size in bytes (default: 100MB)"
    )]
    target_size: u64,
}

#[derive(Debug, Deserialize)]
struct DocumentMetadata {
    repo_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Document {
    metadata: DocumentMetadata,
}

#[derive(Debug)]
struct RepoGroup {
    documents: Vec<serde_json::Value>,
    total_size: u64,
}

#[derive(Debug, Clone)]
struct ProcessedDocument {
    json_value: serde_json::Value,
    repo_name: String,
    size: u64,
}

#[derive(Debug)]
struct OutputBatch {
    output_path: PathBuf,
    documents: Vec<serde_json::Value>,
}


impl RepoGroup {
    fn new(_repo_name: String) -> Self {
        Self {
            documents: Vec::new(),
            total_size: 0,
        }
    }
    
    fn add_document(&mut self, doc_json: serde_json::Value, doc_size: u64) {
        self.documents.push(doc_json);
        self.total_size += doc_size;
    }
    
    fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

fn load_documents_from_file(file_path: &Path, progress_bar: Option<&ProgressBar>) -> Result<Vec<ProcessedDocument>> {
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    
    let mut documents = Vec::new();
    
    if file_path.extension().and_then(|s| s.to_str()) == Some("zst") {
        let decoder = zstd::stream::read::Decoder::new(file)?;
        let reader = std::io::BufReader::new(decoder);
        
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            let doc_size = line.len() as u64;
            let json_value: serde_json::Value = serde_json::from_str(&line)?;
            
            let doc: Document = serde_json::from_value(json_value.clone())?;
            let repo_name = doc.metadata.repo_name.unwrap_or_else(|| "unknown".to_string());
            
            documents.push(ProcessedDocument {
                json_value,
                repo_name,
                size: doc_size,
            });
            
            if let Some(pb) = progress_bar {
                pb.inc(1);
            }
        }
    } else {
        let reader = std::io::BufReader::new(file);
        
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            let doc_size = line.len() as u64;
            let json_value: serde_json::Value = serde_json::from_str(&line)?;
            
            let doc: Document = serde_json::from_value(json_value.clone())?;
            let repo_name = doc.metadata.repo_name.unwrap_or_else(|| "unknown".to_string());
            
            documents.push(ProcessedDocument {
                json_value,
                repo_name,
                size: doc_size,
            });
            
            if let Some(pb) = progress_bar {
                pb.inc(1);
            }
        }
    }
    
    Ok(documents)
}

fn collect_input_files_by_subdir(input_dir: &Path, progress_bar: &ProgressBar) -> Result<HashMap<PathBuf, Vec<PathBuf>>> {
    let mut subdirs: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    
    for entry in WalkDir::new(input_dir) {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            let ext = path.extension().and_then(|s| s.to_str());
            if matches!(ext, Some("jsonl") | Some("zst")) {
                let relative_path = path.strip_prefix(input_dir)?;
                
                let subdir = if let Some(parent) = relative_path.parent() {
                    parent.to_path_buf()
                } else {
                    PathBuf::from(".")
                };
                
                subdirs.entry(subdir).or_insert_with(Vec::new).push(path.to_path_buf());
                progress_bar.inc(1);
            }
        }
    }
    
    Ok(subdirs)
}

fn group_documents_by_repo(documents: Vec<ProcessedDocument>) -> HashMap<String, RepoGroup> {
    documents
        .into_par_iter()
        .fold(
            HashMap::new,
            |mut acc: HashMap<String, Vec<ProcessedDocument>>, doc| {
                acc.entry(doc.repo_name.clone()).or_insert_with(Vec::new).push(doc);
                acc
            },
        )
        .reduce(
            HashMap::new,
            |mut acc, mut map| {
                for (repo_name, docs) in map.drain() {
                    acc.entry(repo_name).or_insert_with(Vec::new).extend(docs);
                }
                acc
            },
        )
        .into_iter()
        .map(|(repo_name, docs)| {
            let mut group = RepoGroup::new(repo_name.clone());
            for doc in docs {
                group.add_document(doc.json_value, doc.size);
            }
            (repo_name, group)
        })
        .collect()
}

fn write_output_file(
    output_path: &Path,
    documents: &[serde_json::Value],
) -> Result<()> {
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path.display()))?;
    
    let buf_writer = BufWriter::new(file);
    let mut encoder = ZstdEncoder::new(buf_writer, 3)?;
    
    for doc in documents {
        serde_json::to_writer(&mut encoder, doc)?;
        encoder.write_all(b"\n")?;
    }
    
    encoder.finish()?;
    Ok(())
}

fn prepare_output_batches(
    repo_groups: HashMap<String, RepoGroup>,
    subdir_output_path: &Path,
    target_size: u64,
) -> Vec<OutputBatch> {
    let mut batches = Vec::new();
    let mut file_counter = 0;
    let mut current_file_docs = Vec::new();
    let mut current_file_size = 0u64;
    
    let mut sorted_repos: Vec<_> = repo_groups.into_iter().collect();
    sorted_repos.sort_by(|a, b| a.0.cmp(&b.0));
    
    for (_repo_name, repo_group) in sorted_repos {
        if repo_group.is_empty() {
            continue;
        }
        
        let repo_would_exceed = current_file_size + repo_group.total_size > target_size;
        let file_not_empty = !current_file_docs.is_empty();
        
        if repo_would_exceed && file_not_empty {
            let output_path = subdir_output_path.join(format!("grouped_{:04}.jsonl.zst", file_counter));
            batches.push(OutputBatch {
                output_path,
                documents: std::mem::take(&mut current_file_docs),
            });
            
            current_file_size = 0;
            file_counter += 1;
        }
        
        current_file_docs.extend(repo_group.documents);
        current_file_size += repo_group.total_size;
    }
    
    if !current_file_docs.is_empty() {
        let output_path = subdir_output_path.join(format!("grouped_{:04}.jsonl.zst", file_counter));
        batches.push(OutputBatch {
            output_path,
            documents: current_file_docs,
        });
    }
    
    batches
}

fn write_output_batches_parallel(batches: Vec<OutputBatch>, progress_bar: &ProgressBar) -> Result<()> {
    let results: Result<Vec<_>> = batches
        .into_par_iter()
        .map(|batch| {
            std::fs::create_dir_all(batch.output_path.parent().unwrap())?;
            info!(
                "Writing file {} with {} documents",
                batch.output_path.display(),
                batch.documents.len()
            );
            let result = write_output_file(&batch.output_path, &batch.documents);
            progress_bar.inc(1);
            result
        })
        .collect();
    
    results.map(|_| ())
}

fn main() -> Result<()> {
    env_logger::init();
    
    let args = Args::parse();
    
    info!("Starting document grouping process");
    info!("Input directory: {}", args.input_dir.display());
    info!("Output directory: {}", args.output_dir.display());
    info!("Target file size: {} bytes", args.target_size);
    
    // Setup progress tracking
    let multi_progress = Arc::new(MultiProgress::new());
    
    // Phase 1: File discovery
    let discovery_pb = multi_progress.add(ProgressBar::new_spinner());
    discovery_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .unwrap()
    );
    discovery_pb.set_message("Discovering input files...");
    
    let subdirs_with_files = collect_input_files_by_subdir(&args.input_dir, &discovery_pb)?;
    let total_files: usize = subdirs_with_files.values().map(|files| files.len()).sum();
    
    discovery_pb.finish_with_message(format!(
        "Found {} files across {} subdirectories", 
        total_files, 
        subdirs_with_files.len()
    ));
    
    if subdirs_with_files.is_empty() {
        warn!("No input files found");
        return Ok(());
    }
    
    // Phase 2: Overall progress tracking
    let overall_pb = multi_progress.add(ProgressBar::new(subdirs_with_files.len() as u64));
    overall_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} subdirs ({percent}%) {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ")
    );
    overall_pb.set_message("Processing subdirectories...");
    
    // Phase 3: Document processing progress
    let doc_processing_pb = multi_progress.add(ProgressBar::new(0));
    doc_processing_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.blue} [{elapsed_precise}] [{bar:40.yellow/red}] {pos} documents processed {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ")
    );
    
    // Count total estimated documents (rough estimate based on files)
    let estimated_docs = total_files * 1000; // rough estimate
    doc_processing_pb.set_length(estimated_docs as u64);
    doc_processing_pb.set_message("Loading and processing documents...");
    
    // Process subdirectories in parallel
    let all_batches: Result<Vec<Vec<OutputBatch>>> = subdirs_with_files
        .into_par_iter()
        .map(|(subdir_path, input_files)| -> Result<Vec<OutputBatch>> {
            info!(
                "Processing subdirectory: {} with {} files", 
                subdir_path.display(), 
                input_files.len()
            );
            
            // Parallel file loading with document progress
            let subdir_documents: Vec<ProcessedDocument> = input_files
                .par_iter()
                .map(|file_path| {
                    info!("Processing file: {}", file_path.display());
                    load_documents_from_file(file_path, Some(&doc_processing_pb))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            
            info!("Loaded {} documents from subdirectory {}", subdir_documents.len(), subdir_path.display());
            
            if subdir_documents.is_empty() {
                overall_pb.inc(1);
                return Ok(Vec::new());
            }
            
            // Parallel grouping
            let repo_groups = group_documents_by_repo(subdir_documents);
            info!("Grouped documents into {} repositories for {}", repo_groups.len(), subdir_path.display());
            
            let subdir_output_path = args.output_dir.join(&subdir_path);
            let batches = prepare_output_batches(repo_groups, &subdir_output_path, args.target_size);
            
            info!("Prepared {} output batches for subdirectory: {}", batches.len(), subdir_path.display());
            overall_pb.inc(1);
            Ok(batches)
        })
        .collect::<Result<Vec<_>>>();
    
    let all_batches = all_batches?;
    
    overall_pb.finish_with_message("All subdirectories processed");
    doc_processing_pb.finish_with_message("All documents loaded and grouped");
    
    // Phase 4: File writing progress
    let all_batches: Vec<OutputBatch> = all_batches.into_iter().flatten().collect();
    let write_pb = multi_progress.add(ProgressBar::new(all_batches.len() as u64));
    write_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} files written ({percent}%) {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ")
    );
    write_pb.set_message("Writing output files...");
    
    info!("Writing {} total output files in parallel", all_batches.len());
    
    write_output_batches_parallel(all_batches, &write_pb)?;
    
    write_pb.finish_with_message("All output files written");
    
    // Final completion message
    let completion_pb = multi_progress.add(ProgressBar::new(1));
    completion_pb.set_style(
        ProgressStyle::default_bar()
            .template("✅ {msg}")
            .unwrap()
    );
    completion_pb.inc(1);
    completion_pb.finish_with_message("Document grouping completed successfully!");
    
    info!("Document grouping completed successfully");
    Ok(())
}