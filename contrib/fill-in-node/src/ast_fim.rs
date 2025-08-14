use rand::prelude::*;
use rand::rng;
use tree_sitter::{Language, Parser as TreeParser, Node};
use std::collections::HashMap;

pub struct AstFillInMiddle<'a> {
    pub fim_rate: f32,
    pub psm_spm_split: f32,
    pub file_separator_token: &'a str,
    pub fim_prefix_token: &'a str,
    pub fim_middle_token: &'a str,
    pub fim_suffix_token: &'a str,
    pub ast_node_distribution: &'a str,
    parsers: HashMap<String, TreeParser>,
}

impl<'a> AstFillInMiddle<'a> {
    pub fn new(
        fim_rate: f32,
        psm_spm_split: f32,
        file_separator_token: &'a str,
        fim_prefix_token: &'a str,
        fim_middle_token: &'a str,
        fim_suffix_token: &'a str,
        ast_node_distribution: &'a str,
    ) -> Self {
        Self {
            fim_rate,
            psm_spm_split,
            file_separator_token,
            fim_prefix_token,
            fim_middle_token,
            fim_suffix_token,
            ast_node_distribution,
            parsers: HashMap::new(),
        }
    }

    pub fn perform_on_document_text_with_replacement(&mut self, document_text: &str, file_separator_replacement: Option<&str>) -> String {
        let mut random = rng();

        let processed_parts: Vec<String> = document_text
            .split(self.file_separator_token)
            .map(|file_text| {
                // Decide whether we're applying FIM to this file text
                if random.random::<f32>() < self.fim_rate {
                    // Try to detect language and apply AST-based FIM
                    if let Some(language) = detect_language_from_content(file_text) {
                        match self.apply_ast_fim(file_text, &language, &mut random) {
                            Ok(transformed) => transformed,
                            Err(_) => {
                                // Fallback to character-level FIM if AST parsing fails
                                self.apply_character_fim(file_text, &mut random)
                            }
                        }
                    } else {
                        // Fallback to character-level FIM for unsupported languages
                        self.apply_character_fim(file_text, &mut random)
                    }
                } else {
                    file_text.to_string()
                }
            })
            .collect();

        // Join the parts with appropriate separator  
        let separator = if let Some(replacement) = file_separator_replacement {
            // Create separator with actual newlines 
            format!("\n{}\n", replacement)
        } else {
            self.file_separator_token.to_string()
        };
        
        processed_parts.join(&separator)
    }

    fn apply_ast_fim(&mut self, file_text: &str, language: &str, random: &mut ThreadRng) -> Result<String, Box<dyn std::error::Error>> {
        // Get or create parser for this language
        let parser = self.parsers.entry(language.to_owned()).or_insert_with(|| {
            let lang = get_language_for_name(language).unwrap();
            let mut parser = TreeParser::new();
            parser.set_language(&lang).unwrap();
            parser
        });
        
        let tree = parser.parse(file_text, None).ok_or("Failed to parse")?;
        let root_node = tree.root_node();
        
        // Collect all suitable nodes for FIM
        let suitable_nodes = collect_suitable_nodes(root_node, file_text.as_bytes());
        
        if suitable_nodes.len() < 3 {
            // Not enough nodes for meaningful FIM, fall back to character-level
            return Ok(self.apply_character_fim(file_text, random));
        }
        
        let (prefix, middle, suffix) = if self.ast_node_distribution == "balanced" {
            self.apply_balanced_node_distribution(&suitable_nodes, file_text, random)?
        } else {
            // Default "single" node strategy
            self.apply_single_node_strategy(&suitable_nodes, file_text, random)?
        };
        
        // Format according to PSM/SPM split
        if random.random::<f32>() < self.psm_spm_split {
            // Prefix-Suffix-Middle
            Ok(format!(
                "{}{}{}{}{}{}",
                self.fim_prefix_token,
                prefix,
                self.fim_suffix_token,
                suffix,
                self.fim_middle_token,
                middle
            ))
        } else {
            // Suffix-Prefix-Middle
            Ok(format!(
                "{}{}{}{}{}{}",
                self.fim_suffix_token,
                suffix,
                self.fim_prefix_token,
                prefix,
                self.fim_middle_token,
                middle
            ))
        }
    }
    
    fn apply_single_node_strategy(&self, suitable_nodes: &[Node], file_text: &str, random: &mut ThreadRng) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        // Select a random node as the "middle" part  
        let middle_idx = random.random_range(1..suitable_nodes.len()-1);
        let middle_node = suitable_nodes[middle_idx];
        
        // Split the text based on the selected node
        let middle_start = middle_node.start_byte();
        let middle_end = middle_node.end_byte();
        
        let prefix = file_text[..middle_start].to_string();
        let middle = file_text[middle_start..middle_end].to_string();
        let suffix = file_text[middle_end..].to_string();
        
        Ok((prefix, middle, suffix))
    }
    
    fn apply_balanced_node_distribution(&self, suitable_nodes: &[Node], file_text: &str, random: &mut ThreadRng) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        // Need at least 3 nodes for balanced distribution
        if suitable_nodes.len() < 3 {
            return Err("Need at least 3 nodes for balanced distribution".into());
        }
        
        let total_bytes = file_text.len();
        let target_prefix_bytes = total_bytes / 3;
        let target_middle_bytes = total_bytes / 3;
        
        // Find suitable split points that create roughly equal byte distributions
        let mut best_prefix_split = 0;
        let mut best_middle_split = 0;
        let mut best_score = f32::INFINITY;
        
        // Try different combinations of split points
        for i in 1..suitable_nodes.len()-1 {
            for j in i+1..suitable_nodes.len() {
                let prefix_split_byte = suitable_nodes[i].end_byte();
                let middle_split_byte = suitable_nodes[j].end_byte();
                
                // Skip if split points are not in correct order
                if prefix_split_byte >= middle_split_byte {
                    continue;
                }
                
                // Calculate actual byte counts for each section
                let prefix_bytes = prefix_split_byte;
                let middle_bytes = middle_split_byte - prefix_split_byte;
                let suffix_bytes = total_bytes - middle_split_byte;
                
                // Calculate deviation from ideal 1/3 distribution
                let prefix_deviation = (prefix_bytes as f32 - target_prefix_bytes as f32).abs();
                let middle_deviation = (middle_bytes as f32 - target_middle_bytes as f32).abs();
                let suffix_deviation = (suffix_bytes as f32 - target_middle_bytes as f32).abs();
                
                let score = prefix_deviation + middle_deviation + suffix_deviation;
                
                if score < best_score {
                    best_score = score;
                    best_prefix_split = i;
                    best_middle_split = j;
                }
            }
        }
        
        // If we didn't find a good combination, fall back to random selection
        // but ensure we have at least some content in each section
        if best_score == f32::INFINITY {
            // Select split points that ensure non-empty sections and proper ordering
            let quarter = suitable_nodes.len() / 4;
            for _ in 0..100 { // Try up to 100 times to find valid splits
                let candidate_prefix = random.random_range(quarter.max(1)..suitable_nodes.len() - quarter.max(1));
                let candidate_middle = random.random_range(candidate_prefix + 1..suitable_nodes.len());
                
                let candidate_prefix_byte = suitable_nodes[candidate_prefix].end_byte();
                let candidate_middle_byte = suitable_nodes[candidate_middle].end_byte();
                
                if candidate_prefix_byte < candidate_middle_byte {
                    best_prefix_split = candidate_prefix;
                    best_middle_split = candidate_middle;
                    break;
                }
            }
            
            // Final fallback: if we still can't find valid splits, return an error
            if best_score == f32::INFINITY {
                return Err("Could not find valid split points with proper byte ordering".into());
            }
        }
        
        let prefix_split_byte = suitable_nodes[best_prefix_split].end_byte();
        let middle_split_byte = suitable_nodes[best_middle_split].end_byte();
        
        // Final validation to ensure proper byte ordering
        if prefix_split_byte >= middle_split_byte {
            return Err("Invalid split points: prefix split byte must be less than middle split byte".into());
        }
        
        let prefix = file_text[..prefix_split_byte].to_string();
        let middle = file_text[prefix_split_byte..middle_split_byte].to_string();
        let suffix = file_text[middle_split_byte..].to_string();
        
        Ok((prefix, middle, suffix))
    }

    // Fallback character-level FIM (similar to original implementation)
    fn apply_character_fim(&mut self, file_text: &str, random: &mut ThreadRng) -> String {
        let file_chars: Vec<char> = file_text.chars().collect();

        // Exclude front and rear character indices we don't want to split at
        let front_offset = 1;
        let rear_offset = 1;
        let range_clip = front_offset + rear_offset + 1;

        // Boundary condition: text is too short to rearrange
        if range_clip > file_chars.len() || (file_chars.len() - range_clip) < 2 {
            return file_text.to_string();
        }

        let mut break_points: Vec<usize> = (0..2)
            .map(|_| random.random_range(front_offset..file_chars.len() - rear_offset))
            .collect();
        break_points.sort();
        break_points.dedup();

        // Ensure we have exactly 2 distinct break points
        while break_points.len() < 2 {
            let new_point = random.random_range(front_offset..file_chars.len() - rear_offset);
            if !break_points.contains(&new_point) {
                break_points.push(new_point);
            }
        }
        break_points.sort();

        // Slice out the chars and back to utf-8 strings
        let prefix = file_chars[..break_points[0]].iter().collect::<String>();
        let middle = file_chars[break_points[0]..break_points[1]]
            .iter()
            .collect::<String>();
        let suffix = file_chars[break_points[1]..].iter().collect::<String>();

        if random.random::<f32>() < self.psm_spm_split {
            // Reorder into Prefix-Suffix-Middle
            format!(
                "{}{}{}{}{}{}",
                self.fim_prefix_token,
                prefix,
                self.fim_suffix_token,
                suffix,
                self.fim_middle_token,
                middle
            )
        } else {
            // Reorder into Suffix-Prefix-Middle
            format!(
                "{}{}{}{}{}{}",
                self.fim_suffix_token,
                suffix,
                self.fim_prefix_token,
                prefix,
                self.fim_middle_token,
                middle
            )
        }
    }
}

fn detect_language_from_content(content: &str) -> Option<&'static str> {
    // Optimized language detection with early returns and &str
    if content.len() > 1000 {
        // Find a safe UTF-8 character boundary at or before index 1000
        let mut safe_index = 1000.min(content.len());
        while safe_index > 0 && !content.is_char_boundary(safe_index) {
            safe_index -= 1;
        }
        let prefix = &content[..safe_index];
        return detect_language_from_prefix(prefix);
    }
    detect_language_from_prefix(content)
}

fn detect_language_from_prefix(content: &str) -> Option<&'static str> {
    // Check for distinctive patterns first (more specific patterns first)
    if content.contains("import java") || content.contains("public class ") {
        return Some("java");
    }
    if content.contains("interface ") || content.contains(": string") {
        return Some("typescript");
    }
    if content.contains("#include") || content.contains("int main") {
        return Some("cpp");
    }
    if content.contains("using ") && content.contains("namespace ") {
        return Some("csharp");
    }
    
    // Check for common keywords
    if content.contains("def ") || content.contains("import ") {
        Some("python")
    } else if content.contains("fn ") || content.contains("use ") {
        Some("rust")
    } else if content.contains("function ") || content.contains("const ") || content.contains("let ") {
        Some("javascript")
    } else if content.contains("package ") || content.contains("func ") {
        Some("go")
    } else {
        None
    }
}

fn get_language_for_name(lang_name: &str) -> Result<Language, Box<dyn std::error::Error>> {
    match lang_name.to_lowercase().as_str() {
        "python" => Ok(tree_sitter_python::language()),
        "rust" => Ok(tree_sitter_rust::language()),
        "cpp" | "c++" => Ok(tree_sitter_cpp::language()),
        "typescript" => Ok(tree_sitter_typescript::language_typescript()),
        "javascript" => Ok(tree_sitter_javascript::language()),
        "java" => Ok(tree_sitter_java::language()),
        "go" => Ok(tree_sitter_go::language()),
        "csharp" | "c#" => Ok(tree_sitter_c_sharp::language()),
        _ => Err(format!("Unsupported language: {}", lang_name).into()),
    }
}

fn collect_suitable_nodes<'a>(node: Node<'a>, source: &'a [u8]) -> Vec<Node<'a>> {
    let mut nodes = Vec::new();
    let mut stack = Vec::new();
    stack.push(node);
    
    while let Some(current_node) = stack.pop() {
        // Only consider nodes that represent complete constructs
        if is_suitable_for_fim(current_node, source) {
            nodes.push(current_node);
        }
        
        // Add children to stack for iterative traversal
        for i in 0..current_node.child_count() {
            if let Some(child) = current_node.child(i) {
                stack.push(child);
            }
        }
    }
    
    nodes
}

fn is_suitable_for_fim(node: Node, _source: &[u8]) -> bool {
    let kind = node.kind();
    let byte_range = node.end_byte() - node.start_byte();
    
    // Skip very small nodes (less than 10 bytes)
    if byte_range < 10 {
        return false;
    }
    
    // Skip comment and string nodes
    if kind.contains("comment") || kind.contains("string") {
        return false;
    }
    
    // Include common statement/expression types that make good FIM candidates
    match kind {
        // Function-like constructs
        "function_definition" | "function_declaration" | "function_item" | "method_declaration" | "method_definition" => true,
        
        // Class/struct/type definitions
        "class_definition" | "class_declaration" | "class_specifier" | "struct_item" | "impl_item" | "type_declaration" => true,
        
        // Control flow statements
        "if_statement" | "if_expression" => true,
        "for_statement" | "for_expression" => true,
        "while_statement" | "while_expression" => true,
        "match_expression" => true,
        
        // Block constructs
        "block" | "compound_statement" => true,
        
        // Other statement types
        "with_statement" | "try_statement" | "expression_statement" => true,
        
        // General statement/expression/declaration pattern (catch remaining valid types)
        _ if kind.ends_with("_statement") || kind.ends_with("_expression") || kind.ends_with("_declaration") => true,
        
        _ => false,
    }
}