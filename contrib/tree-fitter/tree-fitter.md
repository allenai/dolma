# Tree-Fitter Dependency Graph Implementation

## Overview

The tree-fitter program is a sophisticated code analysis tool that processes JSONL.zst files containing code repositories. This document provides a detailed analysis of how the dependency graph implementation works.

## DependencyGraph Data Structure (lines 93-171)

### Core Structure
```rust
#[derive(Debug, Clone)]
struct DependencyGraph {
    nodes: HashSet<String>,           // All file paths in the repository
    edges: HashMap<String, HashSet<String>>, // Adjacency list: file -> files that depend on it
    in_degree: HashMap<String, usize>,        // Track incoming edges for topological sort
}
```

The graph uses three synchronized data structures:
- **nodes**: Contains all unique file paths
- **edges**: Maps each file to the set of files that depend on it (outgoing edges)
- **in_degree**: Tracks how many files each file depends on (incoming edges)

### Key Methods

**`add_node(node: String)`** (lines 109-119):
```rust
fn add_node(&mut self, node: String) {
    use std::collections::hash_map::Entry;
    match self.edges.entry(node.clone()) {
        Entry::Vacant(e) => {
            e.insert(HashSet::new());     // Initialize empty dependency list
            self.nodes.insert(node.clone());
            self.in_degree.insert(node, 0); // No incoming dependencies initially
        }
        Entry::Occupied(_) => {} // Node already exists
    }
}
```

**`add_edge(from: String, to: String)`** (lines 121-132):
```rust
fn add_edge(&mut self, from: String, to: String) {
    self.add_node(from.clone());
    self.add_node(to.clone());
    
    if let Some(edges) = self.edges.get_mut(&from) {
        if edges.insert(to.clone()) {  // Only if edge doesn't already exist
            if let Some(in_degree) = self.in_degree.get_mut(&to) {
                *in_degree += 1;  // Increment incoming edge count
            }
        }
    }
}
```

**Edge Direction**: `from -> to` means "`from` is a dependency of `to`" (i.e., `to` imports/depends on `from`)

## Building the Dependency Graph (lines 463-501)

```rust
fn build_dependency_graph(
    documents: &[Document],
    language: &str,
    doc_map: &HashMap<String, &Document>,
) -> Result<DependencyGraph>
```

### Process Flow:

1. **Initialize**: Create empty graph and language processor
2. **Per Document Processing**:
   ```rust
   for doc in documents {
       if let Some(current_path) = &doc.metadata.path {
           graph.add_node(current_path.clone());  // Add file as node
           
           // Extract imports from source code using tree-sitter
           let imports = processor.extract_imports(&doc.text, Some(current_path))?;
           
           for import in imports {
               if import.import_type == ImportType::Local {  // Only process local imports
                   // Resolve import path to actual file path
                   let dep_path = match language.to_lowercase().as_str() {
                       "python" => find_python_dependency_path(&import.module_path, doc_map, current_path),
                       _ => find_local_dependency(&import.module_path, doc_map, current_path)
                           .map(|doc| doc.metadata.path.as_ref().unwrap().clone()),
                   };
                   
                   if let Some(dep_path) = dep_path {
                       // Add edge: dependency -> current_file
                       graph.add_edge(dep_path, current_path.clone());
                   }
               }
           }
       }
   }
   ```

3. **Import Resolution Examples**:
   - **Python**: `import utils` → looks for `utils.py` or `utils/__init__.py`
   - **Rust**: `mod utils` → looks for `utils.rs` or `utils/mod.rs`
   - **JavaScript**: `import './utils'` → looks for `utils.js`

## Topological Sort Algorithm (lines 134-170)

The implementation uses **Kahn's Algorithm** for topological sorting:

```rust
fn topological_sort(&self) -> Result<Vec<String>> {
    let mut in_degree = self.in_degree.clone();  // Working copy of in-degrees
    let mut queue = VecDeque::new();             // Queue for nodes with no dependencies
    let mut result = Vec::with_capacity(self.nodes.len());

    // Phase 1: Find all nodes with no incoming edges (no dependencies)
    for (node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node.clone());
        }
    }

    // Phase 2: Process nodes in dependency order
    while let Some(node) = queue.pop_front() {
        result.push(node.clone());  // Add to result (safe to process)

        // Remove this node's outgoing edges
        if let Some(neighbors) = self.edges.get(&node) {
            for neighbor in neighbors {
                if let Some(degree) = in_degree.get_mut(neighbor) {
                    *degree -= 1;  // Reduce dependency count
                    if *degree == 0 {  // No more dependencies
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
    }

    // Phase 3: Cycle detection
    if result.len() != self.nodes.len() {
        let result_set: HashSet<String> = result.iter().cloned().collect();
        let remaining: Vec<_> = self.nodes.difference(&result_set).collect();
        return Err(anyhow::anyhow!(
            "Circular dependency detected among: {:?}",
            remaining
        ));
    }

    Ok(result)  // Returns files in dependency order
}
```

### Algorithm Steps:
1. **Initialize**: Copy in-degree counts and create empty queue
2. **Find Roots**: Add all files with zero dependencies to queue
3. **Process**: While queue not empty:
   - Remove a file with no dependencies
   - Add it to result (safe to include in output)
   - Reduce dependency count for all files that depend on it
   - If any dependent file now has zero dependencies, add to queue
4. **Verify**: Check if all files were processed (cycle detection)

## Import Extraction and Resolution Flow

### Tree-sitter Parsing Process (lines 248-293)

```rust
fn extract_imports(&mut self, source: &str, current_path: Option<&str>) -> Result<Vec<ImportInfo>> {
    // 1. Parse source code into AST
    let tree = parser.parse(source, None).context("Failed to parse source code")?;
    
    // 2. Create language-specific query for import statements
    let query = tree_sitter::Query::new(language, &self.import_query)?;
    
    // 3. Execute query against AST
    let matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    
    // 4. Extract import text from matched nodes
    for match_ in matches {
        for capture in match_.captures {
            let import_text = capture.node.utf8_text(source.as_bytes())?;
            let import_info = self.parse_import(import_text, current_path)?;
            imports.push(import_info);
        }
    }
}
```

### Language-Specific Import Queries

**Python** (lines 347-352):
```rust
const PYTHON_IMPORT_QUERY: &str = r#"
(import_statement
  name: (dotted_name) @import)      // import os, sys
(import_from_statement
  module_name: (dotted_name) @import)  // from utils import func
"#;
```

**Rust** (lines 354-359):
```rust
const RUST_IMPORT_QUERY: &str = r#"
(use_declaration
  argument: (scoped_identifier) @import)  // use std::collections::HashMap
(use_declaration
  argument: (identifier) @import)         // use HashMap
"#;
```

### Dependency Resolution Examples

**Example Repository Structure**:
```
repo/
├── main.py          (imports utils, config)
├── utils.py         (imports helpers)
├── helpers.py       (no imports)
└── config.py        (no imports)
```

**Dependency Graph Construction**:
1. **Parse main.py**: Find imports `utils`, `config`
   - Add edges: `utils.py -> main.py`, `config.py -> main.py`
2. **Parse utils.py**: Find import `helpers`
   - Add edge: `helpers.py -> utils.py`
3. **Parse helpers.py, config.py**: No local imports

**Resulting Graph**:
```
nodes: {main.py, utils.py, helpers.py, config.py}
edges: {
  helpers.py -> {utils.py},
  utils.py -> {main.py},
  config.py -> {main.py}
}
in_degree: {
  main.py: 2,      // depends on utils.py, config.py
  utils.py: 1,     // depends on helpers.py
  helpers.py: 0,   // no dependencies
  config.py: 0     // no dependencies
}
```

**Topological Sort Result**: `[helpers.py, config.py, utils.py, main.py]`

This ensures dependencies are processed before files that import them, creating properly ordered concatenated output where all imported code appears before the code that uses it.

### Circular Dependency Detection

If the graph contained a cycle (e.g., `A imports B, B imports C, C imports A`), the topological sort would detect it because not all nodes would be processed - some would remain with non-zero in-degrees, indicating unresolvable circular dependencies.

## Summary

The dependency graph implementation provides:

1. **Accurate Import Analysis**: Uses tree-sitter for robust parsing of import statements
2. **Proper Ordering**: Topological sort ensures dependencies come before dependents
3. **Cycle Detection**: Identifies circular dependencies that would break the ordering
4. **Language Support**: Handles multiple programming languages with specific import patterns
5. **Fallback Mechanisms**: Falls back to simple ordering when dependency analysis fails

This ensures that concatenated repository files maintain proper dependency relationships and can be processed correctly by downstream tools.