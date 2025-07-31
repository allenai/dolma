use clap::Parser;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use tree_sitter::{Language, Parser as TreeParser};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory containing JSONL.zst files organized by language
    #[arg(short, long)]
    input_dir: PathBuf,
    
    /// Output directory for processed files
    #[arg(short, long)]
    output_dir: PathBuf,
    
    /// Include external dependencies in resolution
    #[arg(long, default_value_t = false)]
    include_external: bool,
    
    /// Value of the file separator token
    #[arg(long, default_value = "<|file_sep|>")]
    file_separator_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    added: String,
    created: String,
    id: String,
    metadata: Metadata,
    source: String,
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Metadata {
    detected_licenses: Vec<String>,
    files_concatenated: u32,
    int_score: u32,
    language: String,
    length_bytes: u64,
    license_type: String,
    path: String,
    repo_name: String,
    score: f64,
    src_encoding: String,
    uri: String,
}

#[derive(Debug, Clone)]
struct ImportInfo {
    module_path: String,
    import_type: ImportType,
    line_number: usize,
}

#[derive(Debug, Clone)]
enum ImportType {
    Standard,
    Local,
    External,
}

#[derive(Debug)]
struct ProcessingStats {
    original_length: u64,
    processed_length: u64,
    files_processed: usize,
    imports_resolved: usize,
}

impl ProcessingStats {
    fn new() -> Self {
        Self {
            original_length: 0,
            processed_length: 0,
            files_processed: 0,
            imports_resolved: 0,
        }
    }
    
    fn add(&mut self, other: &ProcessingStats) {
        self.original_length += other.original_length;
        self.processed_length += other.processed_length;
        self.files_processed += other.files_processed;
        self.imports_resolved += other.imports_resolved;
    }
    
    fn average_change(&self) -> f64 {
        if self.files_processed == 0 {
            0.0
        } else {
            (self.processed_length as f64 - self.original_length as f64) / self.files_processed as f64
        }
    }
}

struct LanguageProcessor {
    parser: TreeParser,
    language: Language,
    import_query: String,
}

impl LanguageProcessor {
    fn new(lang: &str) -> Result<Self> {
        let mut parser = TreeParser::new();
        let (language, import_query) = match lang.to_lowercase().as_str() {
            "python" => (tree_sitter_python::language(), PYTHON_IMPORT_QUERY),
            "rust" => (tree_sitter_rust::language(), RUST_IMPORT_QUERY),
            "cpp" | "c++" => (tree_sitter_cpp::language(), CPP_IMPORT_QUERY),
            "typescript" => (tree_sitter_typescript::language_typescript(), TS_IMPORT_QUERY),
            "javascript" => (tree_sitter_javascript::language(), JS_IMPORT_QUERY),
            "java" => (tree_sitter_java::language(), JAVA_IMPORT_QUERY),
            "sql" => return Err(anyhow::anyhow!("SQL import resolution not supported")),  // SQL doesn't have imports
            "c#" | "csharp" => (tree_sitter_c_sharp::language(), CSHARP_IMPORT_QUERY),
            "go" => (tree_sitter_go::language(), GO_IMPORT_QUERY),
            _ => return Err(anyhow::anyhow!("Unsupported language: {}", lang)),
        };
        
        parser.set_language(&language)
            .map_err(|e| anyhow::anyhow!("Failed to set language: {}", e))?;
        
        Ok(Self {
            parser,
            language,
            import_query: import_query.to_string(),
        })
    }
    
    fn extract_imports(&mut self, source: &str) -> Result<Vec<ImportInfo>> {
        let tree = self.parser.parse(source, None)
            .context("Failed to parse source code")?;
        
        let mut imports = Vec::new();
        let mut cursor = tree_sitter::QueryCursor::new();
        let query = tree_sitter::Query::new(&self.language, &self.import_query)
            .map_err(|e| anyhow::anyhow!("Failed to create query: {}", e))?;
        
        let matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
        
        for match_ in matches {
            for capture in match_.captures {
                let node = capture.node;
                if let Ok(import_text) = node.utf8_text(source.as_bytes()) {
                    let import_info = self.parse_import(import_text, node.start_position().row)?;
                    imports.push(import_info);
                }
            }
        }
        
        Ok(imports)
    }
    
    fn parse_import(&self, import_text: &str, line_number: usize) -> Result<ImportInfo> {
        // This is a simplified import parsing - would need language-specific logic
        let import_type = if import_text.contains("std") || import_text.contains("os") {
            ImportType::Standard
        } else if import_text.starts_with('.') || import_text.contains("./") {
            ImportType::Local
        } else {
            ImportType::External
        };
        
        Ok(ImportInfo {
            module_path: import_text.to_string(),
            import_type,
            line_number,
        })
    }
}

// Tree-sitter query strings for different languages
const PYTHON_IMPORT_QUERY: &str = r#"
(import_statement
  name: (dotted_name) @import)
(import_from_statement
  module_name: (dotted_name) @import)
"#;

const RUST_IMPORT_QUERY: &str = r#"
(use_declaration
  argument: (scoped_identifier) @import)
(use_declaration
  argument: (identifier) @import)
"#;

const CPP_IMPORT_QUERY: &str = r#"
(preproc_include
  path: (string_literal) @import)
(preproc_include
  path: (system_lib_string) @import)
"#;

const TS_IMPORT_QUERY: &str = r#"
(import_statement
  source: (string) @import)
"#;

const JS_IMPORT_QUERY: &str = r#"
(import_statement
  source: (string) @import)
"#;

const JAVA_IMPORT_QUERY: &str = r#"
(import_declaration
  (scoped_identifier) @import)
"#;

const SQL_IMPORT_QUERY: &str = r#""#; // SQL doesn't typically have imports

const CSHARP_IMPORT_QUERY: &str = r#"
(using_directive
  name: (identifier) @import)
(using_directive
  name: (qualified_name) @import)
"#;

const GO_IMPORT_QUERY: &str = r#"
(import_spec
  path: (interpreted_string_literal) @import)
"#;

fn read_jsonl_zst_file(file_path: &Path) -> Result<Vec<Document>> {
    let file = std::fs::File::open(file_path)?;
    let decoder = zstd::Decoder::new(file)?;
    let reader = std::io::BufReader::new(decoder);
    
    let mut documents = Vec::new();
    for line in std::io::BufRead::lines(reader) {
        let line = line?;
        if !line.trim().is_empty() {
            let doc: Document = serde_json::from_str(&line)?;
            documents.push(doc);
        }
    }
    
    Ok(documents)
}

fn write_jsonl_zst_file(documents: &[Document], output_path: &Path) -> Result<()> {
    std::fs::create_dir_all(output_path.parent().unwrap())?;
    let file = std::fs::File::create(output_path)?;
    let encoder = zstd::Encoder::new(file, 0)?;
    let mut writer = std::io::BufWriter::new(encoder.auto_finish());
    
    for doc in documents {
        let json = serde_json::to_string(doc)?;
        std::io::Write::write_all(&mut writer, json.as_bytes())?;
        std::io::Write::write_all(&mut writer, b"\n")?;
    }
    
    Ok(())
}

fn group_documents_by_repo(documents: Vec<Document>) -> HashMap<String, Vec<Document>> {
    let mut grouped = HashMap::new();
    
    for doc in documents {
        let key = format!("{}_{}", doc.metadata.repo_name, doc.metadata.language);
        grouped.entry(key).or_insert_with(Vec::new).push(doc);
    }
    
    grouped
}

fn resolve_dependencies(
    documents: &[Document],
    language: &str,
    include_external: bool,
    file_separator_token: &str,
) -> Result<(Vec<Document>, ProcessingStats)> {
    let mut processor = LanguageProcessor::new(language)?;
    let mut stats = ProcessingStats::new();
    
    // Create a map of file paths to documents for quick lookup
    let doc_map: HashMap<String, &Document> = documents
        .iter()
        .map(|doc| (doc.metadata.path.clone(), doc))
        .collect();
    
    let mut resolved_docs = Vec::new();
    
    for doc in documents {
        stats.original_length += doc.text.len() as u64;
        stats.files_processed += 1;
        
        let imports = processor.extract_imports(&doc.text)?;
        let mut resolved_text = doc.text.clone();
        
        for import in imports {
            match import.import_type {
                ImportType::Local => {
                    if let Some(dep_doc) = find_local_dependency(&import.module_path, &doc_map, &doc.metadata.path) {
                        resolved_text.push_str(file_separator_token);
                        resolved_text.push_str(&dep_doc.text);
                        stats.imports_resolved += 1;
                    }
                }
                ImportType::External if include_external => {
                    // For external dependencies, we would need additional logic
                    // to fetch from package managers, but for now we'll skip
                }
                _ => {}
            }
        }
        
        stats.processed_length += resolved_text.len() as u64;
        
        let mut new_doc = doc.clone();
        new_doc.text = resolved_text;
        resolved_docs.push(new_doc);
    }
    
    Ok((resolved_docs, stats))
}

fn find_local_dependency<'a>(
    import_path: &str,
    doc_map: &'a HashMap<String, &'a Document>,
    current_path: &str,
) -> Option<&'a Document> {
    // Simplified local dependency resolution
    // In a real implementation, this would need language-specific path resolution logic
    
    let current_dir = Path::new(current_path).parent()?;
    let potential_paths = vec![
        format!("{}/{}.py", current_dir.display(), import_path),
        format!("{}/{}/mod.rs", current_dir.display(), import_path),
        format!("{}/{}.rs", current_dir.display(), import_path),
        format!("{}/{}.js", current_dir.display(), import_path),
        format!("{}/{}.ts", current_dir.display(), import_path),
    ];
    
    for path in potential_paths {
        if let Some(doc) = doc_map.get(&path) {
            return Some(doc);
        }
    }
    
    None
}

fn concatenate_repo_files(documents: Vec<Document>, file_separator_token: &str) -> Document {
    if documents.is_empty() {
        panic!("Cannot concatenate empty document list");
    }
    
    let first_doc = &documents[0];
    let document_texts: Vec<String> = documents.iter().map(|doc| doc.text.clone()).collect();
    let concatenated_text = document_texts.join(file_separator_token);
    let total_length: u64 = documents.iter().map(|doc| doc.metadata.length_bytes).sum();
    
    let mut result = first_doc.clone();
    result.text = concatenated_text;
    result.metadata.length_bytes = total_length;
    result.metadata.files_concatenated = documents.len() as u32;
    result.metadata.path = format!("{}_concatenated", first_doc.metadata.repo_name);
    
    result
}

fn process_language_directory(
    input_dir: &Path, 
    output_dir: &Path, 
    language: &str, 
    include_external: bool,
    file_separator_token: &str,
) -> Result<ProcessingStats> {
    let lang_dir = input_dir.join(language);
    if !lang_dir.exists() {
        return Ok(ProcessingStats::new());
    }
    
    let mut total_stats = ProcessingStats::new();
    let files: Vec<_> = WalkDir::new(&lang_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "zst"))
        .collect();
    
    let results: Vec<Result<ProcessingStats>> = files
        .par_iter()
        .map(|entry| {
            let file_path = entry.path();
            println!("Processing: {}", file_path.display());
            
            let documents = read_jsonl_zst_file(file_path)?;
            let grouped = group_documents_by_repo(documents);
            
            let mut file_stats = ProcessingStats::new();
            let mut output_docs = Vec::new();
            
            for (_repo_key, repo_docs) in grouped {
                let (resolved_docs, stats) = resolve_dependencies(&repo_docs, language, include_external, file_separator_token)?;
                file_stats.add(&stats);
                
                let concatenated = concatenate_repo_files(resolved_docs, file_separator_token);
                output_docs.push(concatenated);
            }
            
            let output_file = output_dir
                .join(language)
                .join(file_path.file_name().unwrap());
            write_jsonl_zst_file(&output_docs, &output_file)?;
            
            Ok(file_stats)
        })
        .collect();
    
    for result in results {
        total_stats.add(&result?);
    }
    
    Ok(total_stats)
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    std::fs::create_dir_all(&args.output_dir)?;
    
    let languages = vec![
        "Python", "Rust", "C++", "TypeScript", "JavaScript", 
        "Java", "SQL", "C#", "Go"
    ];
    
    let mut overall_stats = ProcessingStats::new();
    
    println!("Starting parallel processing of language directories...");
    
    let results: Vec<Result<ProcessingStats>> = languages
        .par_iter()
        .map(|&language| {
            println!("Processing language: {}", language);
            process_language_directory(&args.input_dir, &args.output_dir, language, args.include_external, &args.file_separator_token)
        })
        .collect();
    
    for result in results {
        overall_stats.add(&result?);
    }
    
    println!("\n=== Processing Statistics ===");
    println!("Files processed: {}", overall_stats.files_processed);
    println!("Imports resolved: {}", overall_stats.imports_resolved);
    println!("Original total length: {} bytes", overall_stats.original_length);
    println!("Processed total length: {} bytes", overall_stats.processed_length);
    println!("Average change per document: {:.2} bytes", overall_stats.average_change());
    println!("Total size change: {:.2}%", 
        if overall_stats.original_length > 0 {
            ((overall_stats.processed_length as f64 - overall_stats.original_length as f64) / overall_stats.original_length as f64) * 100.0
        } else {
            0.0
        }
    );
    
    Ok(())
}