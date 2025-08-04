use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    detected_licenses: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    int_score: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    length_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    license_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    src_encoding: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    uri: Option<String>,
}

#[derive(Debug, Clone)]
struct ImportInfo {
    module_path: String,
    import_type: ImportType,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone)]
struct DependencyGraph {
    nodes: HashSet<String>,
    edges: HashMap<String, HashSet<String>>,
    in_degree: HashMap<String, usize>,
}

impl DependencyGraph {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: HashSet::with_capacity(capacity),
            edges: HashMap::with_capacity(capacity),
            in_degree: HashMap::with_capacity(capacity),
        }
    }

    fn add_node(&mut self, node: String) {
        use std::collections::hash_map::Entry;
        match self.edges.entry(node.clone()) {
            Entry::Vacant(e) => {
                e.insert(HashSet::new());
                self.nodes.insert(node.clone());
                self.in_degree.insert(node, 0);
            }
            Entry::Occupied(_) => {}
        }
    }

    fn add_edge(&mut self, from: String, to: String) {
        self.add_node(from.clone());
        self.add_node(to.clone());
        
        if let Some(edges) = self.edges.get_mut(&from) {
            if edges.insert(to.clone()) {
                if let Some(in_degree) = self.in_degree.get_mut(&to) {
                    *in_degree += 1;
                }
            }
        }
    }

    fn topological_sort(&self) -> Result<Vec<String>> {
        let mut in_degree = self.in_degree.clone();
        let mut queue = VecDeque::new();
        let mut result = Vec::with_capacity(self.nodes.len());

        for (node, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node.clone());
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            if let Some(neighbors) = self.edges.get(&node) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            let result_set: HashSet<String> = result.iter().cloned().collect();
            let remaining: Vec<_> = self.nodes.difference(&result_set).collect();
            return Err(anyhow::anyhow!(
                "Circular dependency detected among: {:?}",
                remaining
            ));
        }

        Ok(result)
    }
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
            (self.processed_length as f64 - self.original_length as f64)
                / self.files_processed as f64
        }
    }
}

struct LanguageProcessor {
    parser: Option<TreeParser>,
    language: Option<Language>,
    import_query: String,
    is_supported: bool,
}

impl LanguageProcessor {
    fn new(lang: &str) -> Result<Self> {
        let lang_lower = lang.to_lowercase();
        let (language_opt, import_query, is_supported) = match lang_lower.as_str() {
            "python" => (Some(tree_sitter_python::language()), PYTHON_IMPORT_QUERY, true),
            "rust" => (Some(tree_sitter_rust::language()), RUST_IMPORT_QUERY, true),
            "cpp" | "c++" => (Some(tree_sitter_cpp::language()), CPP_IMPORT_QUERY, true),
            "typescript" => (
                Some(tree_sitter_typescript::language_typescript()),
                TS_IMPORT_QUERY,
                true,
            ),
            "javascript" => (Some(tree_sitter_javascript::language()), JS_IMPORT_QUERY, true),
            "java" => (Some(tree_sitter_java::language()), JAVA_IMPORT_QUERY, true),
            "c#" | "csharp" => (Some(tree_sitter_c_sharp::language()), CSHARP_IMPORT_QUERY, true),
            "go" => (Some(tree_sitter_go::language()), GO_IMPORT_QUERY, true),
            "sql" => {
                warn!("SQL language detected - import resolution not supported, using fallback ordering");
                (None, "", false)
            },
            _ => {
                warn!("Unsupported language '{}' - using fallback ordering", lang);
                (None, "", false)
            },
        };

        let (parser_opt, language_final) = if let Some(language) = language_opt {
            let mut parser = TreeParser::new();
            parser
                .set_language(&language)
                .map_err(|e| anyhow::anyhow!("Failed to set language: {}", e))?;
            (Some(parser), Some(language))
        } else {
            (None, None)
        };

        Ok(Self {
            parser: parser_opt,
            language: language_final,
            import_query: import_query.to_string(),
            is_supported,
        })
    }

    fn extract_imports(&mut self, source: &str, current_path: Option<&str>) -> Result<Vec<ImportInfo>> {
        if !self.is_supported {
            debug!("Language not supported for import extraction, returning empty imports");
            return Ok(Vec::new());
        }

        let parser = self.parser.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Parser not available for unsupported language")
        })?;
        
        let language = self.language.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Language not available for unsupported language")
        })?;

        debug!(
            "Extracting imports from source code ({} bytes)",
            source.len()
        );
        let tree = parser
            .parse(source, None)
            .context("Failed to parse source code")?;

        let mut imports = Vec::new();
        let mut cursor = tree_sitter::QueryCursor::new();
        let query = tree_sitter::Query::new(language, &self.import_query)
            .map_err(|e| anyhow::anyhow!("Failed to create query: {}", e))?;

        let matches = cursor.matches(&query, tree.root_node(), source.as_bytes());

        for match_ in matches {
            for capture in match_.captures {
                let node = capture.node;
                if let Ok(import_text) = node.utf8_text(source.as_bytes()) {
                    let import_info = self.parse_import(import_text, current_path)?;
                    debug!(
                        "Found import: {} (type: {:?})",
                        import_info.module_path, import_info.import_type
                    );
                    imports.push(import_info);
                }
            }
        }

        debug!("Extracted {} imports from source", imports.len());
        Ok(imports)
    }

    fn parse_import(&self, import_text: &str, current_path: Option<&str>) -> Result<ImportInfo> {
        let cleaned_import = import_text.trim_matches('"').trim_matches('\'');
        
        let (import_type, resolved_path) = if let Some(language) = &self.language {
            match language {
            lang if *lang == tree_sitter_python::language() => {
                let import_type = if cleaned_import.starts_with('.') {
                    ImportType::Local
                } else if is_python_standard_library(cleaned_import) {
                    ImportType::Standard
                } else {
                    ImportType::External
                };

                let resolved_path = if import_type == ImportType::Local {
                    if let Some(path) = current_path {
                        resolve_python_relative_import(cleaned_import, path)
                    } else {
                        cleaned_import.to_string()
                    }
                } else {
                    cleaned_import.to_string()
                };

                (import_type, resolved_path)
            }
            _ => {
                // Generic logic for other languages
                let import_type = if cleaned_import.contains("std") || cleaned_import.contains("os") {
                    ImportType::Standard
                } else if cleaned_import.starts_with('.') || cleaned_import.contains("./") {
                    ImportType::Local
                } else {
                    ImportType::External
                };
                (import_type, cleaned_import.to_string())
            }
        }
        } else {
            // Fallback for unsupported languages
            (ImportType::External, cleaned_import.to_string())
        };

        Ok(ImportInfo {
            module_path: resolved_path,
            import_type,
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

fn is_python_standard_library(module_name: &str) -> bool {
    let stdlib_modules = [
        "os", "sys", "json", "re", "math", "random", "datetime", "collections",
        "itertools", "functools", "pathlib", "typing", "ast", "copy", "pickle",
        "subprocess", "threading", "multiprocessing", "asyncio", "urllib", "http",
        "email", "html", "xml", "csv", "sqlite3", "logging", "unittest", "argparse",
        "configparser", "io", "tempfile", "shutil", "glob", "fnmatch", "linecache",
        "platform", "getpass", "time", "calendar", "hashlib", "hmac", "secrets",
        "uuid", "base64", "binascii", "struct", "codecs", "locale", "gettext",
        "decimal", "fractions", "statistics", "array", "weakref", "types", "gc",
        "inspect", "site", "importlib", "pkgutil", "modulefinder", "runpy",
        "warnings", "contextlib", "abc", "atexit", "traceback", "signal",
        "socket", "ssl", "select", "selectors", "queue", "sched", "heapq",
        "bisect", "pprint", "reprlib", "enum", "graphlib", "dataclasses",
    ];
    
    let root_module = module_name.split('.').next().unwrap_or(module_name);
    stdlib_modules.contains(&root_module)
}

fn resolve_python_relative_import(import_path: &str, current_file_path: &str) -> String {
    let current_dir = Path::new(current_file_path).parent().unwrap_or(Path::new(""));
    
    if import_path.starts_with('.') {
        let level = import_path.chars().take_while(|&c| c == '.').count();
        let module_part = &import_path[level..];
        
        let mut target_dir = current_dir;
        for _ in 1..level {
            target_dir = target_dir.parent().unwrap_or(Path::new(""));
        }
        
        if module_part.is_empty() {
            format!("{}", target_dir.display())
        } else {
            format!("{}/{}", target_dir.display(), module_part.replace('.', "/"))
        }
    } else {
        import_path.replace('.', "/")
    }
}

fn simple_file_ordering(documents: &[Document]) -> Vec<&Document> {
    info!("Using simple file ordering (no dependency analysis) for {} documents", documents.len());
    
    // Simple heuristic: sort by path length (shorter paths first, likely dependencies)
    // then alphabetically for deterministic ordering
    let mut docs_with_paths: Vec<_> = documents.iter().collect();
    docs_with_paths.sort_by(|a, b| {
        let path_a = a.metadata.path.as_deref().unwrap_or("");
        let path_b = b.metadata.path.as_deref().unwrap_or("");
        
        // First sort by path depth (fewer slashes = likely higher in hierarchy)
        let depth_a = path_a.matches('/').count();
        let depth_b = path_b.matches('/').count();
        
        match depth_a.cmp(&depth_b) {
            std::cmp::Ordering::Equal => path_a.cmp(path_b), // Then alphabetically
            other => other,
        }
    });
    
    info!("Simple file ordering complete - {} documents sorted by path hierarchy", docs_with_paths.len());
    docs_with_paths
}

fn build_dependency_graph(
    documents: &[Document],
    language: &str,
    doc_map: &HashMap<String, &Document>,
) -> Result<DependencyGraph> {
    info!("Building dependency graph for {} documents in {}", documents.len(), language);
    let mut graph = DependencyGraph::with_capacity(documents.len());
    let mut processor = LanguageProcessor::new(language)?;
    
    if !processor.is_supported {
        return Err(anyhow::anyhow!("Language {} not supported for dependency graph building", language));
    }

    for doc in documents {
        if let Some(current_path) = &doc.metadata.path {
            graph.add_node(current_path.clone());
            
            if language.to_lowercase() == "python" {
                let imports = processor.extract_imports(&doc.text, Some(current_path))?;
                
                for import in imports {
                    if import.import_type == ImportType::Local {
                        if let Some(dep_path) = find_python_dependency_path(&import.module_path, doc_map, current_path) {
                            debug!("Adding dependency edge: {} -> {}", current_path, dep_path);
                            graph.add_edge(dep_path, current_path.clone());
                        }
                    }
                }
            }
        }
    }
    
    info!("Dependency graph built with {} nodes", graph.nodes.len());
    Ok(graph)
}

fn find_python_dependency_path(
    import_path: &str,
    doc_map: &HashMap<String, &Document>,
    _current_path: &str,
) -> Option<String> {
    let potential_paths = vec![
        format!("{}.py", import_path),
        format!("{}/__init__.py", import_path),
    ];
    
    for path in &potential_paths {
        if doc_map.contains_key(path) {
            return Some(path.clone());
        }
    }
    
    None
}

fn read_jsonl_zst_file(file_path: &Path) -> Result<Vec<Document>> {
    debug!("Reading JSONL.zst file: {}", file_path.display());
    let file = std::fs::File::open(file_path)?;
    let decoder = zstd::Decoder::new(file)?;
    let reader = std::io::BufReader::new(decoder);

    let mut documents = Vec::new();
    let mut line_count = 0;
    for line in std::io::BufRead::lines(reader) {
        let line = line?;
        line_count += 1;
        if !line.trim().is_empty() {
            let doc: Document = serde_json::from_str(&line).with_context(|| {
                format!(
                    "Failed to parse JSON on line {} in file {}",
                    line_count,
                    file_path.display()
                )
            })?;
            documents.push(doc);
        }
    }

    info!(
        "Loaded {} documents from {} ({} lines processed)",
        documents.len(),
        file_path.display(),
        line_count
    );
    Ok(documents)
}

fn write_jsonl_zst_file(documents: &[Document], output_path: &Path) -> Result<()> {
    debug!(
        "Writing {} documents to {}",
        documents.len(),
        output_path.display()
    );
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(output_path)?;
    let encoder = zstd::Encoder::new(file, 0)?;
    let mut writer = std::io::BufWriter::new(encoder.auto_finish());

    let total_size_before: usize = documents.iter().map(|doc| doc.text.len()).sum();

    for doc in documents {
        let json = serde_json::to_string(doc)?;
        std::io::Write::write_all(&mut writer, json.as_bytes())?;
        std::io::Write::write_all(&mut writer, b"\n")?;
    }

    info!(
        "Successfully wrote {} documents ({} bytes total content) to {}",
        documents.len(),
        total_size_before,
        output_path.display()
    );

    Ok(())
}

fn group_documents_by_repo(documents: Vec<Document>) -> HashMap<String, Vec<Document>> {
    debug!("Grouping {} documents by repository", documents.len());
    let mut grouped = HashMap::with_capacity(documents.len() / 10); // Estimate

    for doc in documents {
        let repo_name = doc.metadata.repo_name.as_deref().unwrap_or("unknown");
        let language = doc.metadata.language.as_deref().unwrap_or("unknown");
        let key = format!("{}_{}", repo_name, language);
        grouped.entry(key).or_insert_with(Vec::new).push(doc);
    }

    info!("Grouped documents into {} repositories", grouped.len());
    for (repo_key, docs) in &grouped {
        debug!("Repository '{}': {} documents", repo_key, docs.len());
    }

    grouped
}

fn resolve_dependencies(
    documents: &[Document],
    language: &str,
    include_external: bool,
    file_separator_token: &str,
) -> Result<(Vec<Document>, ProcessingStats)> {
    info!(
        "Starting dependency resolution for {} documents in {}",
        documents.len(),
        language
    );
    let mut processor = LanguageProcessor::new(language)?;
    let mut stats = ProcessingStats::new();
    let mut total_imports_found = 0;
    let mut imports_by_type = HashMap::new();
    let mut resolution_failures = HashMap::new();

    // Create a map of file paths to documents for quick lookup
    let doc_map: HashMap<String, &Document> = documents
        .iter()
        .filter_map(|doc| doc.metadata.path.as_ref().map(|path| (path.clone(), doc)))
        .collect();

    info!(
        "Created document map with {} entries for dependency lookup",
        doc_map.len()
    );

    // Try to build dependency graph and get topological order for supported languages
    let processing_order = if language.to_lowercase() == "python" {
        match build_dependency_graph(documents, language, &doc_map) {
            Ok(graph) => {
                match graph.topological_sort() {
                    Ok(sorted_paths) => {
                        info!("Successfully computed topological order for {} files", sorted_paths.len());
                        // Create ordered list of documents based on topological sort
                        let mut ordered_docs = Vec::new();
                        let mut processed_paths = HashSet::new();
                        
                        // First, add documents in topological order
                        for path in &sorted_paths {
                            if let Some(doc) = doc_map.get(path) {
                                ordered_docs.push(*doc);
                                processed_paths.insert(path.clone());
                            }
                        }
                        
                        // Add any remaining documents that weren't in the dependency graph
                        for doc in documents {
                            if let Some(path) = &doc.metadata.path {
                                if !processed_paths.contains(path) {
                                    ordered_docs.push(doc);
                                }
                            } else {
                                ordered_docs.push(doc);
                            }
                        }
                        
                        ordered_docs
                    }
                    Err(e) => {
                        warn!("Circular dependency detected: {}. Falling back to simple file ordering.", e);
                        simple_file_ordering(documents)
                    }
                }
            }
            Err(e) => {
                warn!("Failed to build dependency graph: {}. Falling back to simple file ordering.", e);
                simple_file_ordering(documents)
            }
        }
    } else {
        // For non-Python languages or unsupported languages, use simple file ordering
        info!("Using simple file ordering for language: {}", language);
        simple_file_ordering(documents)
    };

    let mut resolved_docs = Vec::new();

    for (idx, doc) in processing_order.iter().enumerate() {
        debug!(
            "Processing document {}/{}: {}",
            idx + 1,
            processing_order.len(),
            doc.metadata.path.as_deref().unwrap_or("<unknown path>")
        );

        stats.original_length += doc.text.len() as u64;
        stats.files_processed += 1;

        let imports = match processor.extract_imports(&doc.text, doc.metadata.path.as_deref()) {
            Ok(imports) => imports,
            Err(e) => {
                warn!("Failed to extract imports for document {}: {}. Continuing without import resolution.", 
                      doc.metadata.path.as_deref().unwrap_or("<unknown>"), e);
                Vec::new()
            }
        };
        let mut resolved_text = doc.text.clone();
        let original_size = resolved_text.len();
        total_imports_found += imports.len();

        if !imports.is_empty() {
            debug!("Found {} imports to resolve", imports.len());
        }

        for import in imports {
            // Track import types
            *imports_by_type.entry(format!("{:?}", import.import_type)).or_insert(0) += 1;
            
            debug!(
                "Resolving import: {} (type: {:?})",
                import.module_path, import.import_type
            );
            match import.import_type {
                ImportType::Local => {
                    if let Some(current_path) = &doc.metadata.path {
                        if let Some(dep_doc) =
                            find_local_dependency(&import.module_path, &doc_map, current_path)
                        {
                            debug!(
                                "Successfully resolved local dependency: {} -> {}",
                                import.module_path,
                                dep_doc.metadata.path.as_deref().unwrap_or("<unknown>")
                            );
                            resolved_text.push_str("\n");
                            resolved_text.push_str(file_separator_token);
                            resolved_text.push_str("\n");
                            resolved_text.push_str(&dep_doc.text);
                            stats.imports_resolved += 1;
                        } else {
                            let failure_key = "Local dependency not found in document map";
                            *resolution_failures.entry(failure_key.to_string()).or_insert(0) += 1;
                            debug!(
                                "Could not resolve local dependency: {} from {} - not found in document map",
                                import.module_path, current_path
                            );
                        }
                    } else {
                        let failure_key = "Document missing path metadata";
                        *resolution_failures.entry(failure_key.to_string()).or_insert(0) += 1;
                        debug!(
                            "Document has no path metadata, cannot resolve local import: {}",
                            import.module_path
                        );
                    }
                }
                ImportType::External if include_external => {
                    let failure_key = "External dependency resolution not implemented";
                    *resolution_failures.entry(failure_key.to_string()).or_insert(0) += 1;
                    debug!(
                        "External dependency resolution not implemented: {}",
                        import.module_path
                    );
                }
                ImportType::External => {
                    let failure_key = "External dependencies disabled (use --include-external)";
                    *resolution_failures.entry(failure_key.to_string()).or_insert(0) += 1;
                    debug!(
                        "Skipping external dependency (--include-external not set): {}",
                        import.module_path
                    );
                }
                ImportType::Standard => {
                    let failure_key = "Standard library imports skipped by design";
                    *resolution_failures.entry(failure_key.to_string()).or_insert(0) += 1;
                    debug!("Skipping standard library import: {}", import.module_path);
                }
            }
        }

        let size_change = resolved_text.len() as i64 - original_size as i64;
        if size_change > 0 {
            debug!(
                "Document size increased by {} bytes after dependency injection",
                size_change
            );
        }

        stats.processed_length += resolved_text.len() as u64;

        let mut new_doc = (*doc).clone();
        new_doc.text = resolved_text;
        resolved_docs.push(new_doc);
    }

    info!(
        "Dependency resolution complete: {}/{} imports resolved, {} documents processed",
        stats.imports_resolved,
        total_imports_found,
        stats.files_processed
    );
    
    // Log detailed breakdown of import types
    info!("Import breakdown by type:");
    for (import_type, count) in &imports_by_type {
        info!("  {}: {} imports", import_type, count);
    }
    
    // Log reasons why imports were not resolved
    if !resolution_failures.is_empty() {
        info!("Reasons imports were not resolved:");
        for (reason, count) in &resolution_failures {
            info!("  {}: {} imports", reason, count);
        }
    }

    Ok((resolved_docs, stats))
}

fn find_local_dependency<'a>(
    import_path: &str,
    doc_map: &'a HashMap<String, &'a Document>,
    current_path: &str,
) -> Option<&'a Document> {
    debug!(
        "Looking for local dependency '{}' from '{}'",
        import_path, current_path
    );

    let current_dir = Path::new(current_path).parent()?;
    let potential_paths = vec![
        format!("{}/{}.py", current_dir.display(), import_path),
        format!("{}/{}/mod.rs", current_dir.display(), import_path),
        format!("{}/{}.rs", current_dir.display(), import_path),
        format!("{}/{}.js", current_dir.display(), import_path),
        format!("{}/{}.ts", current_dir.display(), import_path),
    ];

    debug!(
        "Checking {} potential paths for dependency",
        potential_paths.len()
    );

    for path in &potential_paths {
        debug!("Checking path: {}", path);
        if let Some(doc) = doc_map.get(path) {
            debug!("Found dependency at: {}", path);
            return Some(doc);
        }
    }

    debug!(
        "No dependency found for '{}' in any of the checked paths",
        import_path
    );
    None
}

fn concatenate_repo_files(documents: Vec<Document>, file_separator_token: &str) -> Document {
    if documents.is_empty() {
        panic!("Cannot concatenate empty document list");
    }

    info!(
        "Concatenating {} documents into single repository file",
        documents.len()
    );

    let first_doc = &documents[0];
    let separator = format!("\n{}\n", file_separator_token);
    let mut concatenated_text = String::with_capacity(
        documents.iter().map(|doc| doc.text.len()).sum::<usize>() 
        + (documents.len() - 1) * separator.len()
    );
    
    for (i, doc) in documents.iter().enumerate() {
        if i > 0 {
            concatenated_text.push_str(&separator);
        }
        concatenated_text.push_str(&doc.text);
    }
    let final_size = concatenated_text.len();

    let total_length: u64 = documents
        .iter()
        .filter_map(|doc| doc.metadata.length_bytes)
        .sum();

    let mut result = first_doc.clone();
    result.text = concatenated_text;
    result.metadata.length_bytes = Some(total_length);
    let repo_name = first_doc.metadata.repo_name.as_deref().unwrap_or("unknown");
    result.metadata.path = Some(format!("{}_concatenated", repo_name));

    info!(
        "Concatenation complete: {} bytes total content (+{} bytes delta)",
        final_size,
        final_size as i64 - total_length as i64
    );

    result
}

fn create_progress_bar(file_count: usize, language: &str) -> ProgressBar {
    let pb = ProgressBar::new(file_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .expect("Invalid progress bar template")
            .progress_chars("##-"),
    );
    pb.set_message(format!("Processing {} files", language));
    pb
}

fn process_single_file(
    file_path: &Path,
    input_base_dir: &Path,
    output_dir: &Path,
    language: &str,
    include_external: bool,
    file_separator_token: &str,
    preserve_structure: bool,
    pb: &Arc<Mutex<ProgressBar>>,
) -> Result<ProcessingStats> {
    debug!("Processing file: {}", file_path.display());
    let documents = read_jsonl_zst_file(file_path)?;
    let grouped = group_documents_by_repo(documents);

    let mut file_stats = ProcessingStats::new();
    let mut output_docs = Vec::new();

    for (repo_key, repo_docs) in grouped {
        info!(
            "Processing repository: {} ({} documents)",
            repo_key,
            repo_docs.len()
        );
        let (resolved_docs, stats) = resolve_dependencies(
            &repo_docs,
            language,
            include_external,
            file_separator_token,
        )?;
        file_stats.add(&stats);

        let concatenated = concatenate_repo_files(resolved_docs, file_separator_token);
        output_docs.push(concatenated);
        info!("Completed processing repository: {}", repo_key);
    }

    let output_file = if preserve_structure {
        let relative_path = file_path.strip_prefix(input_base_dir)?;
        output_dir.join(relative_path)
    } else {
        output_dir
            .join(language)
            .join(file_path.file_name().ok_or_else(|| {
                anyhow::anyhow!("File path has no filename: {}", file_path.display())
            })?)
    };
    
    write_jsonl_zst_file(&output_docs, &output_file)?;

    if let Ok(pb) = pb.lock() {
        pb.inc(1);
        pb.set_message(format!(
            "Processing {} - {}",
            language,
            file_path.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_else(|| "unknown".into())
        ));
    }

    Ok(file_stats)
}

fn process_files_in_directory(
    input_dir: &Path,
    output_dir: &Path,
    language: &str,
    include_external: bool,
    file_separator_token: &str,
    preserve_structure: bool,
) -> Result<ProcessingStats> {
    if !input_dir.exists() {
        return Ok(ProcessingStats::new());
    }

    let files: Vec<_> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "zst"))
        .collect();

    if files.is_empty() {
        println!("No .zst files found in {}", input_dir.display());
        return Ok(ProcessingStats::new());
    }

    let pb = create_progress_bar(files.len(), language);
    let pb = Arc::new(Mutex::new(pb));

    let results: Vec<Result<ProcessingStats>> = files
        .par_iter()
        .map(|entry| {
            process_single_file(
                entry.path(),
                input_dir,
                output_dir,
                language,
                include_external,
                file_separator_token,
                preserve_structure,
                &pb,
            )
        })
        .collect();

    if let Ok(pb) = pb.lock() {
        pb.finish_with_message(format!("Completed {} ({} files)", language, files.len()));
    }

    let mut total_stats = ProcessingStats::new();
    for result in results {
        total_stats.add(&result?);
    }

    Ok(total_stats)
}


fn detect_language_from_path(input_dir: &Path) -> Option<String> {
    if let Some(dir_name) = input_dir.file_name() {
        let dir_str = dir_name.to_string_lossy().to_lowercase();
        match dir_str.as_str() {
            "python" => Some("Python".to_string()),
            "rust" => Some("Rust".to_string()),
            "cpp" | "c++" => Some("C++".to_string()),
            "typescript" => Some("TypeScript".to_string()),
            "javascript" => Some("JavaScript".to_string()),
            "java" => Some("Java".to_string()),
            "sql" => Some("SQL".to_string()),
            "c#" | "csharp" => Some("C#".to_string()),
            "go" => Some("Go".to_string()),
            _ => None,
        }
    } else {
        None
    }
}

fn process_detected_language(
    args: &Args,
    detected_language: &str,
) -> Result<ProcessingStats> {
    info!("Processing language: {}", detected_language);

    let stats = process_files_in_directory(
        &args.input_dir,
        &args.output_dir,
        detected_language,
        args.include_external,
        &args.file_separator_token,
        true,
    )?;
    
    info!("Completed processing for detected language: {}", detected_language);
    Ok(stats)
}

fn process_multiple_languages(args: &Args) -> Result<ProcessingStats> {
    let languages = &[
        "Python", "Rust", "C++", "TypeScript", "JavaScript", 
        "Java", "SQL", "C#", "Go",
    ];

    info!("Starting parallel processing of language directories...");

    let existing_languages: Vec<&str> = languages
        .iter()
        .filter(|&&lang| {
            let lang_dir = args.input_dir.join(lang);
            let exists = lang_dir.exists();
            if exists {
                debug!("Found language directory: {}", lang_dir.display());
            } else {
                debug!("Language directory not found: {}", lang_dir.display());
            }
            exists
        })
        .copied()
        .collect();

    if existing_languages.is_empty() {
        warn!("No language directories found in {}", args.input_dir.display());
        return Ok(ProcessingStats::new());
    }

    info!(
        "Found {} language directories to process: {:?}",
        existing_languages.len(),
        existing_languages
    );

    let results: Vec<Result<ProcessingStats>> = existing_languages
        .par_iter()
        .map(|&language| {
            let lang_dir = args.input_dir.join(language);
            process_files_in_directory(
                &lang_dir,
                &args.output_dir,
                language,
                args.include_external,
                &args.file_separator_token,
                false,
            )
        })
        .collect();

    let mut overall_stats = ProcessingStats::new();
    for result in results {
        overall_stats.add(&result?);
    }
    
    info!("Completed parallel processing of all language directories");
    Ok(overall_stats)
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    let args = Args::parse();

    info!("Starting tree-fitter processing");
    info!("Input directory: {}", args.input_dir.display());
    info!("Output directory: {}", args.output_dir.display());
    info!("Include external dependencies: {}", args.include_external);
    info!("File separator token: '{}'", args.file_separator_token);

    std::fs::create_dir_all(&args.output_dir)?;

    let overall_stats = if let Some(detected_language) = detect_language_from_path(&args.input_dir) {
        info!("Detected language from input directory: {}", detected_language);
        process_detected_language(&args, &detected_language)?
    } else {
        info!("No specific language detected, processing all language directories");
        process_multiple_languages(&args)?
    };

    info!("\n=== Processing Statistics ===");
    info!("Files processed: {}", overall_stats.files_processed);
    info!("Imports resolved: {}", overall_stats.imports_resolved);
    info!(
        "Original total length: {} bytes",
        overall_stats.original_length
    );
    info!(
        "Processed total length: {} bytes",
        overall_stats.processed_length
    );
    info!(
        "Average change per document: {:.2} bytes",
        overall_stats.average_change()
    );
    let size_change_percent = if overall_stats.original_length > 0 {
        ((overall_stats.processed_length as f64 - overall_stats.original_length as f64)
            / overall_stats.original_length as f64)
            * 100.0
    } else {
        0.0
    };
    info!("Total size change: {:.2}%", size_change_percent);

    if overall_stats.imports_resolved > 0 {
        info!(
            "Successfully injected {} dependencies across {} files",
            overall_stats.imports_resolved, overall_stats.files_processed
        );
    } else {
        info!("No dependencies were resolved and injected");
    }

    info!("Tree-fitter processing completed successfully");

    Ok(())
}
