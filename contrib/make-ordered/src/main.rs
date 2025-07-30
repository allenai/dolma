use anyhow::{Context, Result};
use clap::Parser;
use log::{info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
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
    #[serde(flatten)]
    other: serde_json::Value,
}

#[derive(Debug)]
struct RepoGroup {
    repo_name: String,
    documents: Vec<serde_json::Value>,
    total_size: u64,
}

#[derive(Debug)]
struct SubdirectoryData {
    relative_path: PathBuf,
    repo_groups: HashMap<String, RepoGroup>,
}

impl RepoGroup {
    fn new(repo_name: String) -> Self {
        Self {
            repo_name,
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

fn load_documents_from_file(file_path: &Path) -> Result<Vec<(serde_json::Value, String, u64)>> {
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
            
            documents.push((json_value, repo_name, doc_size));
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
            
            documents.push((json_value, repo_name, doc_size));
        }
    }
    
    Ok(documents)
}

fn collect_input_files_by_subdir(input_dir: &Path) -> Result<HashMap<PathBuf, Vec<PathBuf>>> {
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
            }
        }
    }
    
    Ok(subdirs)
}

fn group_documents_by_repo(documents: Vec<(serde_json::Value, String, u64)>) -> HashMap<String, RepoGroup> {
    let mut repo_groups: HashMap<String, RepoGroup> = HashMap::new();
    
    for (doc_json, repo_name, doc_size) in documents {
        let group = repo_groups.entry(repo_name.clone()).or_insert_with(|| RepoGroup::new(repo_name));
        group.add_document(doc_json, doc_size);
    }
    
    repo_groups
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

fn write_grouped_files_for_subdir(
    repo_groups: HashMap<String, RepoGroup>,
    subdir_output_path: &Path,
    target_size: u64,
) -> Result<()> {
    std::fs::create_dir_all(subdir_output_path)?;
    
    let mut file_counter = 0;
    let mut current_file_docs = Vec::new();
    let mut current_file_size = 0u64;
    let mut current_repos = Vec::new();
    
    let mut sorted_repos: Vec<_> = repo_groups.into_iter().collect();
    sorted_repos.sort_by(|a, b| a.0.cmp(&b.0));
    
    for (repo_name, repo_group) in sorted_repos {
        if repo_group.is_empty() {
            continue;
        }
        
        let repo_would_exceed = current_file_size + repo_group.total_size > target_size;
        let file_not_empty = !current_file_docs.is_empty();
        
        if repo_would_exceed && file_not_empty {
            let output_path = subdir_output_path.join(format!("grouped_{:04}.jsonl.zst", file_counter));
            info!(
                "Writing file {} with {} documents from repos: {:?}",
                output_path.display(),
                current_file_docs.len(),
                current_repos
            );
            
            write_output_file(&output_path, &current_file_docs)?;
            
            current_file_docs.clear();
            current_file_size = 0;
            current_repos.clear();
            file_counter += 1;
        }
        
        current_file_docs.extend(repo_group.documents);
        current_file_size += repo_group.total_size;
        current_repos.push(repo_name);
    }
    
    if !current_file_docs.is_empty() {
        let output_path = subdir_output_path.join(format!("grouped_{:04}.jsonl.zst", file_counter));
        info!(
            "Writing final file {} with {} documents from repos: {:?}",
            output_path.display(),
            current_file_docs.len(),
            current_repos
        );
        
        write_output_file(&output_path, &current_file_docs)?;
    }
    
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    
    let args = Args::parse();
    
    info!("Starting document grouping process");
    info!("Input directory: {}", args.input_dir.display());
    info!("Output directory: {}", args.output_dir.display());
    info!("Target file size: {} bytes", args.target_size);
    
    let subdirs_with_files = collect_input_files_by_subdir(&args.input_dir)?;
    info!("Found {} subdirectories with files", subdirs_with_files.len());
    
    if subdirs_with_files.is_empty() {
        warn!("No input files found");
        return Ok(());
    }
    
    for (subdir_path, input_files) in subdirs_with_files {
        info!(
            "Processing subdirectory: {} with {} files", 
            subdir_path.display(), 
            input_files.len()
        );
        
        let subdir_documents: Vec<(serde_json::Value, String, u64)> = input_files
            .par_iter()
            .map(|file_path| {
                info!("Processing file: {}", file_path.display());
                load_documents_from_file(file_path)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        
        info!("Loaded {} documents from subdirectory {}", subdir_documents.len(), subdir_path.display());
        
        if subdir_documents.is_empty() {
            continue;
        }
        
        let repo_groups = group_documents_by_repo(subdir_documents);
        info!("Grouped documents into {} repositories for {}", repo_groups.len(), subdir_path.display());
        
        let subdir_output_path = args.output_dir.join(&subdir_path);
        write_grouped_files_for_subdir(repo_groups, &subdir_output_path, args.target_size)?;
        
        info!("Completed processing subdirectory: {}", subdir_path.display());
    }
    
    info!("Document grouping completed successfully");
    Ok(())
}